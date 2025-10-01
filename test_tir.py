from __future__ import annotations
import os, json, argparse
from typing import List, Dict
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import T5Tokenizer, T5Config
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode



# --- utils ---------------------------------------------------------------
from utils_tir import (
    T5Dataset, T5ForPAGSeqIdGeneration,
    Trie, prefix_allowed_tokens_fn,
    load_codes, PAGLogitsProcessor, PrefixPriorLogitsProcessor
)

def compute_recall_mrr(preds: List[str], gts: List[str], top_k=50):
    n = len(preds) // top_k #preds= n * top_k
    R1 = R5 = R10 = 0.0
    for i in range(n):
        beams = [preds[i * top_k + j].split() for j in range(top_k)]
        label = gts[i].split()
        try:
            r = next(j for j, b in enumerate(beams) if b == label)
        except StopIteration:
            continue
        if r < 1:  R1 += 1
        if r < 5:  R5 += 1
        if r < 10: R10 += 1
    return {"Recall@1": R1 / n, "Recall@5": R5 / n, "Recall@10": R10 / n}




def collate(batch, pad_id):
    ids = torch.stack([x["input_ids"] for x in batch])
    lbl = torch.stack([x["labels"] for x in batch])
    mask = (ids != pad_id).long()
    return {"input_ids": ids, "attention_mask": mask, "labels": lbl}

@torch.no_grad()
def evaluate(
    model, tok, ds,
    img_vecs: torch.Tensor, cap_vecs: torch.Tensor, seq2row: Dict[str,int],
    *, rerank: bool,
    data_path: str,
    beam=50, topk=50, max_new=8,
    λ_token=0.5, λ_prefix=1.0,
    c_start=32100, code_sz=1024,
    n_candidates=500, τ=8.0,
    device="cuda",
    blip_rerank: bool = False,
    blip_model=None,
    image_names=None,
    blip_image_dir=None,
    captions=None,
    use_trie=True,
):
    from PIL import Image

    # -------blip -------
    image_size = 384
    blip_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])
    # ------- blip --------


    print("n_candidates :", n_candidates)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=lambda b: collate(b, tok.pad_token_id))
    model = model.to(device).eval()
    if img_vecs is not None:
        img_vecs = img_vecs.to(device)
    if cap_vecs is not None:
        cap_vecs = cap_vecs.to(device)

    total_gen_time = 0.0
    total_queries  = 0

    tgt_path = os.path.join(data_path, "test.target")
    seq_lists = [[int(t.lstrip("c_")) for t in line.split()]
                 for line in open(tgt_path).read().splitlines()]
    m = max(len(s) for s in seq_lists)
    for s in seq_lists: s.extend([0]*(m-len(s)))
    seq_ids_tensor = torch.tensor(seq_lists, device=device)

    if use_trie:
        trie = Trie([[tok.pad_token_id] + tok.encode(c) for c in load_codes(tgt_path)])
        prefix_fn = prefix_allowed_tokens_fn(trie)
    else:
        prefix_fn = None

    gen_cfg = dict(num_beams=beam, num_return_sequences=topk,
                   max_new_tokens=max_new, early_stopping=True, use_cache=True)

    all_preds, global_idx = [], 0

    for batch in tqdm(loader, desc="Eval"):
        batch = {k: v.to(device) for k, v in batch.items()}
        ids, mask = batch["input_ids"], batch["attention_mask"]
        dec_start = model.config.decoder_start_token_id or tok.pad_token_id
        dummy_dec = torch.full((ids.size(0), 1), dec_start, device=device)

        out = model(input_ids=ids, attention_mask=mask,
                    decoder_input_ids=dummy_dec, return_dict=True)
        h_q = out.h_q_seq
        B = h_q.size(0)

        # ---- logits processors ------------------------------------------
        lp = []
        if λ_token > 0:
            lp.append(PAGLogitsProcessor(
                h_q.repeat_interleave(beam, 0), λ_token, c_start, code_sz))
        if λ_prefix > 0:
            scores_tok = (h_q.unsqueeze(1).expand(-1, seq_ids_tensor.size(0), -1)
                          .gather(2, seq_ids_tensor.unsqueeze(0).expand(B, -1, -1)))
            s_global = scores_tok.sum(-1)
            top_val, top_idx = torch.topk(s_global, k=n_candidates, dim=-1)
            prefix_priors = []
            for b in range(B):
                D = {}
                for j, ix in enumerate(top_idx[b]):
                    raw = top_val[b, j].item()
                    seq = seq_ids_tensor[ix].tolist()
                    m_eff = len([x for x in seq if x])
                    g = raw / (m_eff * τ)
                    for t in range(1, m_eff + 1):
                        pref = (dec_start,) + tuple(x + c_start for x in seq[:t])
                        D[pref] = max(D.get(pref, 0.0), g)
                prefix_priors.append(D)
            lp.append(PrefixPriorLogitsProcessor(
                h_q.repeat_interleave(beam, 0), λ_prefix, c_start,
                code_sz, prefix_priors, beam))

        # beams = model.generate(
        #     input_ids=ids, attention_mask=mask,
        #     logits_processor=lp or None,
        #     prefix_allowed_tokens_fn=prefix_fn,
        #     **gen_cfg
        # )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        # [B * topk, seq_len]
        beams = model.generate(
            input_ids=ids, attention_mask=mask,
            logits_processor=lp or None,
            prefix_allowed_tokens_fn=prefix_fn, **gen_cfg
        )

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        total_gen_time += time.perf_counter() - t0
        total_queries  += B

        # ---------- rerank  ----------------------------------------------
        if rerank:
            # beams_ = beams.view(B, topk, -1)#[batch_size, topk, seq_len]

            seq_keys = []
            for flat_k in range(B * topk):
                toks = []
                for tid in beams[flat_k]:
                    if tid in (tok.pad_token_id, tok.eos_token_id):
                        continue
                    dec = tok.decode([tid], skip_special_tokens=False)
                    if dec.startswith("c_"):
                        toks.append(dec)
                        if len(toks) == 4:
                            break
                seq_keys.append(" ".join(toks))#seq_keys [c_4 c_8 c_22 c_18]

            row_idx = torch.tensor([seq2row.get(k, -1) for k in seq_keys],
                                   device=device)
            valid = row_idx >= 0
            row_idx_clamped = row_idx.clamp_min(0)

            if blip_rerank:
                # BLIP ITM rerank
                need_idx = torch.unique(row_idx_clamped).tolist()  #
                img_bank = {}  # {img_idx: tensor}
                for idx in need_idx:
                    img_path = os.path.join(blip_image_dir, image_names[idx])
                    img_bank[idx] = blip_transform(
                        Image.open(img_path).convert("RGB")
                    ).to(device, non_blocking=True)  # (3, H, W)
                blip_itm_scores = torch.zeros(B, topk)
                for b in range(B):
                    cap_idx = global_idx + b
                    caption = captions[cap_idx]
                    for k in range(topk):
                        img_idx = row_idx_clamped[b * topk + k].item()
                        img_tensor = img_bank[img_idx].unsqueeze(0)  # (1,3,H,W)

                        with torch.no_grad():
                            itm_output = blip_model(img_tensor, caption, match_head='itm')
                            itm_score = torch.nn.functional.softmax(itm_output, dim=1)[:, 1].item()
                        blip_itm_scores[b, k] = itm_score
                sort_idx = blip_itm_scores.argsort(dim=-1, descending=True)

            #clip-rerank
            else:
                cand_img = img_vecs[row_idx_clamped]
                cand_cap = cap_vecs[row_idx_clamped]
                cand_vec = F.normalize((cand_img + cand_cap) / 2, dim=-1)
                cand_vec[~valid] = 0
                q_vec = cap_vecs[global_idx: global_idx + B].to(device)
                sims = (q_vec.repeat_interleave(topk, 0) * cand_vec).sum(-1)
                sims = sims.view(B, topk)
                sort_idx = sims.argsort(dim=-1, descending=True) #[batch_size, topk]

            # selected = torch.gather(beams_, 1, sort_idx.to(beams_.device).unsqueeze(-1).expand_as(beams_)).reshape(-1, beams.size(-1))
            #beams_   [batch_size, topk, seq_len] -> [batch_size * topk, seq_len]     #remark
            sort_idx = sort_idx.to(device)  #
            sort_idx_flat = (torch.arange(B, device=device).unsqueeze(1) * topk + sort_idx).view(-1)#(B,topk)

            selected = beams.index_select(0, sort_idx_flat)  # (B*topk, seq_len)
            global_idx += B

            #beams_ = [
            #[ [c_9, c_12, c_27, c_30],  [c_4, c_8, c_22, c_18],  [c_11, c_6, c_3, c_5] ],  # topk=3

        else:
            selected = beams
            global_idx += B

        all_preds.extend(tok.batch_decode(selected, skip_special_tokens=True))#

    gts = [tok.decode([t for t in l if t >= 0], skip_special_tokens=True).strip()
           for l in ds[:]["labels"]]

    metrics = compute_recall_mrr(all_preds, gts, topk) #[c_9, c_12, c_27, c_30]
    #topk-version
    # eval_topks = [3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # metrics = {}
    # for k in eval_topks:
    #     m = compute_recall_mrr_eval(all_preds, gts, topk, eval_topk=k)
    #     metrics.update(m)

    metrics["Latency(ms/query)"] = total_gen_time / total_queries * 1000
    return metrics

# ---------------- CLI ----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--beam", type=int, default=50)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--lambda_token", type=float, default=0.5)
    parser.add_argument("--lambda_prefix", type=float, default=1.0)
    parser.add_argument("--rerank", action="store_true", help="rerank")
    parser.add_argument("--head_path", type=str, default=None)
    parser.add_argument("--caption_emb_path", required=False, help="Only needed for CLIP/cap embedding rerank")
    parser.add_argument("--image_emb_path", required=False, help="Only needed for CLIP/cap embedding rerank")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_candidates", type=int, default=500)
    parser.add_argument('--codebook', type=int, default=1024)
    parser.add_argument('--max_target_len', type=int, default=6)
    # BLIP rerank
    parser.add_argument("--blip_rerank", action="store_true", help="Use BLIP ITM for rerank")
    parser.add_argument("--blip_model_path", type=str, default=None, help="BLIP ITM model path")
    parser.add_argument("--blip_image_json", type=str, default=None, help="image list json ")
    parser.add_argument("--blip_image_dir", type=str, default=None, help="image file dir")
    parser.add_argument("--blip_caption_path", type=str, default=None, help="caption file path")
    parser.add_argument('--no_trie', action='store_true', help='Do NOT use Trie prefix guidance')



    args = parser.parse_args()

    use_trie = not args.no_trie  # 
    print("⚠️Data path", args.data_path)

    tok = T5Tokenizer.from_pretrained(args.ckpt)
    cfg = T5Config.from_pretrained(args.ckpt)
    cfg.pag_code_book_size = args.codebook
    model = T5ForPAGSeqIdGeneration.from_pretrained(args.ckpt, config=cfg)

    if args.head_path and os.path.exists(args.head_path):
        model.seq_id_preference_head.load_state_dict(
            torch.load(args.head_path, map_location="cpu"))
        print(f"✅ loaded custom head from {args.head_path}")

    ds = T5Dataset(
        tok,
        os.path.join(args.data_path, "test.source"),
        os.path.join(args.data_path, "test.target"),
        max_source_len=128, max_target_len=args.max_target_len
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.rerank and not args.blip_rerank:
        if args.image_emb_path is None or args.caption_emb_path is None:
            raise ValueError("For CLIP/cap rerank, you must provide --image_emb_path and --caption_emb_path")
        img_vecs = F.normalize(torch.from_numpy(
            np.load(args.image_emb_path)).float().to(device), dim=-1)
        cap_vecs = F.normalize(torch.from_numpy(
            np.load(args.caption_emb_path)).float().to(device), dim=-1)
    else:
        img_vecs, cap_vecs = None, None
    tgt_lines = open(os.path.join(args.data_path, "test.target")).read().splitlines()
    seq2row = {" ".join(l.split()): i for i, l in enumerate(tgt_lines)}


    print("rerank or not :", args.rerank)
    print("beam size :", args.beam)
    print("lambda_token :", args.lambda_token)
    print("lambda_prefix :", args.lambda_prefix)
    print("topk :", args.topk)
    print("blip_rerank :", args.blip_rerank)
    print("no_trie :", args.no_trie)

    blip_model, image_names, captions = None, None, None
    import sys

    sys.path.insert(0, "/home/xxx/blip")
    print("sys.path:", sys.path)


    if args.blip_rerank:
        from models.blip_itm import blip_itm

        print("blip_itm loaded!", blip_itm)
        blip_model = blip_itm(
            pretrained=args.blip_model_path,
            image_size=384,
            vit='large'  #
        ).eval().to(device)
        with open(args.blip_image_json, "r") as f:
            image_names = json.load(f)
        with open(args.blip_caption_path, "r") as f:
            captions = [l.strip() for l in f]

    metrics = evaluate(
        model, tok, ds,
        img_vecs=img_vecs, cap_vecs=cap_vecs, seq2row=seq2row,
        rerank=args.rerank,
        beam=args.beam, topk=args.topk,
        λ_token=args.lambda_token, λ_prefix=args.lambda_prefix,
        device=device, data_path=args.data_path,
        n_candidates=args.n_candidates,
        blip_rerank=args.blip_rerank,
        blip_model=blip_model,
        image_names=image_names,
        blip_image_dir=args.blip_image_dir,
        captions=captions,
        use_trie=use_trie,
        code_sz=args.codebook
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
