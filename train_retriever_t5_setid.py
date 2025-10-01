#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
import os
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
import wandb
from datetime import datetime
from transformers import (
    T5Config, T5EncoderModel, T5Tokenizer,
    AdamW, get_linear_schedule_with_warmup
)

# ---------------- Model Definition ----------------
class GuidanceScorePretrainer(nn.Module):
    def __init__(self, encoder: T5EncoderModel, d_model: int, codebook_size: int):
        super().__init__()
        self.encoder = encoder
        self.seq_id_preference_head = nn.Linear(d_model, codebook_size)

    def forward(self, input_ids=None, attention_mask=None):
        enc_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        pooled = enc_out.last_hidden_state.max(dim=1)[0]
        return self.seq_id_preference_head(pooled)

    def compute_multi_positive_loss(self, logits, pos_ids):
        log_probs = F.log_softmax(logits, dim=1)
        pos_log_probs = log_probs.gather(1, pos_ids)
        return -pos_log_probs.mean()

# ---------------- Dataset ----------------
class CaptionToImageCodeDataset(Dataset):
    def __init__(self, tokenizer: T5Tokenizer, src_file: str, tgt_file: str, max_len: int = 128):
        self.tokenizer = tokenizer
        with open(src_file, 'r', encoding='utf-8') as f:
            self.captions = [l.strip() for l in f]
        with open(tgt_file, 'r', encoding='utf-8') as f:
            self.labels = [list(map(lambda x: int(x.lstrip('c_')), l.strip().split())) for l in f]
        assert len(self.captions) == len(self.labels), "Caption and label counts do not match."
        self.max_len = max_len

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        tk = self.tokenizer(
            self.captions[idx], max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': tk.input_ids.squeeze(0),
            'attention_mask': tk.attention_mask.squeeze(0),
            'positive_code_ids': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ---------------- Validation & Test ----------------
def evaluate_split(model, dl, device, distributed):
    model.eval()
    total_loss, total_samples = 0.0, 0
    with torch.no_grad():
        for batch in dl:
            ids = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            pos = batch['positive_code_ids'].to(device)
            logits = model(ids, masks)
            loss = (model.module if distributed else model).compute_multi_positive_loss(logits, pos)
            total_loss += loss.item() * ids.size(0)
            total_samples += ids.size(0)
    if distributed:
        stats = torch.tensor([total_loss, total_samples], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss, total_samples = stats.tolist()
    return total_loss / total_samples

def evaluate_test(model, dl, device, distributed):
    model.eval()
    k_list = [1, 5, 10]
    hits_at_k = [0, 0, 0]
    strict4 = 0
    total_loss, total_samples = 0.0, 0
    with torch.no_grad():
        for batch in dl:
            ids = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            pos = batch['positive_code_ids'].to(device)
            logits = model(ids, masks)
            loss = (model.module if distributed else model).compute_multi_positive_loss(logits, pos)
            total_loss += loss.item() * ids.size(0)
            topk = torch.topk(logits, k=k_list[-1], dim=1).indices
            for i in range(topk.size(0)):
                ps = set(pos[i].tolist())
                pr = topk[i].tolist()
                for idx, k in enumerate(k_list):
                    if any(p in ps for p in pr[:k]):
                        hits_at_k[idx] += 1
                if set(pr[:4]).issubset(ps):
                    strict4 += 1
            total_samples += ids.size(0)
    if distributed:
        stats = torch.tensor([total_loss, total_samples, strict4] + hits_at_k,
                             dtype=torch.float64, device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss, total_samples, strict4, *hits_at_k = stats.tolist()
    return total_loss / total_samples, strict4 / total_samples, [h / total_samples for h in hits_at_k]

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--t5_model_name_or_path', required=True)
    parser.add_argument('--caption_file_train', required=True)
    parser.add_argument('--image_code_file_train', required=True)
    parser.add_argument('--caption_file_val', required=True)
    parser.add_argument('--image_code_file_val', required=True)
    parser.add_argument('--caption_file_test', required=True)
    parser.add_argument('--image_code_file_test', required=True)
    parser.add_argument('--codebook_embedding_path', required=True)
    parser.add_argument('--pag_code_book_size', type=int, default=1024)
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_train_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--warmup_steps', type=int, default=10)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--eval_steps', type=int, default=1000)
    parser.add_argument('--wandb_project', type=str, default='Stage1SetID')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # distributed init
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    distributed = int(os.getenv('WORLD_SIZE', 1)) > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl', init_method='env://')
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        rank, world_size = 0, 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # WandB
    if rank == 0:
        wandb.init(
            project=args.wandb_project,
            name=timestamp,  #
            config=vars(args)
        )

    # seed & device
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    if rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # model & tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.t5_model_name_or_path)
    config = T5Config.from_pretrained(args.t5_model_name_or_path)
    encoder = T5EncoderModel.from_pretrained(args.t5_model_name_or_path, config=config)
    model = GuidanceScorePretrainer(encoder, config.d_model, args.pag_code_book_size).to(device)
    cb = torch.load(args.codebook_embedding_path, map_location=device)
    with torch.no_grad():
        model.seq_id_preference_head.weight.copy_(cb)
        if model.seq_id_preference_head.bias is not None:
            model.seq_id_preference_head.bias.zero_()
    if distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # dataloaders
    train_ds = CaptionToImageCodeDataset(tokenizer, args.caption_file_train, args.image_code_file_train, args.max_seq_length)
    val_ds = CaptionToImageCodeDataset(tokenizer, args.caption_file_val, args.image_code_file_val, args.max_seq_length)
    test_ds = CaptionToImageCodeDataset(tokenizer, args.caption_file_test, args.image_code_file_test, args.max_seq_length)
    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None
    test_sampler = DistributedSampler(test_ds, shuffle=False) if distributed else None
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, shuffle=(train_sampler is None))
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, sampler=test_sampler, shuffle=False)

    # optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    total_steps = len(train_dl) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )

    #
    best_val_loss = float('inf')
    best_enc_path, best_head_path = None, None
    global_step = 0
    patience = 2
    steps_without_improve = 0
    early_stop = False

    for epoch in range(1, args.num_train_epochs + 1):
        if early_stop:
            break
        if train_sampler:
            train_sampler.set_epoch(epoch)
        model.train()

        for batch in tqdm(train_dl, desc=f'Epoch {epoch}', disable=(rank != 0)):
            optimizer.zero_grad()
            ids = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            pos = batch['positive_code_ids'].to(device)
            logits = model(ids, masks)
            loss = (model.module if distributed else model).compute_multi_positive_loss(logits, pos)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            # ---------- step-level eval ----------
            if global_step % args.eval_steps == 0:
                val_loss = evaluate_split(model, val_dl, device, distributed)
                test_loss, test_strict, test_rec = evaluate_test(
                    model, test_dl, device, distributed
                )
                if rank == 0:
                    wandb.log({
                        'step_val_loss': val_loss,
                        'step_test_loss': test_loss,
                        'step_strict4': test_strict,
                        'step_recall1': test_rec[0],
                        'step_recall5': test_rec[1],
                        'step_recall10': test_rec[2]
                    }, step=global_step)

                model.train()
                if val_loss < best_val_loss:

                    best_val_loss = val_loss
                    steps_without_improve = 0  #
                    if rank == 0:
                        print(f"[SAVE] step={global_step}  val_loss={val_loss:.4f}")
                        best_enc_path = os.path.join(
                            args.output_dir, f'best_enc_step{global_step}.pt')
                        torch.save((model.module if distributed else model)
                                   .encoder.state_dict(), best_enc_path)
                        best_head_path = os.path.join(
                            args.output_dir, f'best_head_step{global_step}.pt')
                        torch.save((model.module if distributed else model)
                                   .seq_id_preference_head.state_dict(), best_head_path)
                else:
                    steps_without_improve += 1
                    if steps_without_improve >= patience:
                        if rank == 0:
                            print(f"No improvement for {patience} eval-steps — early stop.")
                        early_stop = True
                        break  #


        # epoch end eval & logging
        val_loss = evaluate_split(model, val_dl, device, distributed)
        test_loss, test_strict, test_rec = evaluate_test(model, test_dl, device, distributed)
        if rank == 0:
            wandb.log({
                'epoch': epoch,
                'val_loss': val_loss,
                'test_loss': test_loss,
                'epoch_strict4': test_strict,
                'epoch_recall1': test_rec[0],
                'epoch_recall5': test_rec[1],
                'epoch_recall10': test_rec[2]
            }, step=global_step)
            print(
                f"Epoch {epoch}: Val Loss={val_loss:.4f}, Test Loss={test_loss:.4f}, "
                f"Strict4={test_strict:.4f}, R1={test_rec[0]:.4f}, R5={test_rec[1]:.4f}, R10={test_rec[2]:.4f}"
            )



    if distributed:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
