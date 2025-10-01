
import os
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorWithPadding,
    GenerationConfig,
    T5ForConditionalGeneration,
    T5Config,
)
import torch
import logging
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import random
import wandb
from typing import Dict, List
import torch.nn.functional as F
import sys
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.generation.logits_process import LogitsProcessor
import torch.nn as nn


def preprocess_function(source_text, target_text, tokenizer):
    prefix_text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction:{source_text} ### Response:"
    response_text = f"{prefix_text}{target_text}</s>"

    input = tokenizer(
        response_text,
        return_tensors=None,
        max_length=128,
        truncation=True,
        padding="max_length",
    )
    input_ids = input["input_ids"]

    labels = input_ids.copy()

    output_start_index = (
            len(
                tokenizer.encode(
                    prefix_text, max_length=128, truncation=True, padding="max_length"
                )
            )
            - 9
    )
    labels[:output_start_index] = [-100] * output_start_index

    return {
        "input_ids": input_ids,
        "attention_mask": input["attention_mask"],
        "labels": labels,
    }


def prefix_allowed_tokens_fn(candidate_trie):
    def prefix_allowed_tokens(batch_id, sentence):
        sentence = sentence.tolist()
        trie_out = candidate_trie.get(sentence)
        return trie_out

    return prefix_allowed_tokens


def llama_prefix_allowed_tokens_fn(candidate_trie):
    def prefix_allowed_tokens(batch_id, sentence):
        sentence = sentence.tolist()
        # you may need to change this value in your case
        index = sentence.index(13291)
        sentence = sentence[index:]
        trie_out = candidate_trie.get(sentence)
        return trie_out

    return prefix_allowed_tokens


def load_codes(target_file):
    with open(target_file, "r", encoding="utf-8") as tgt_file:
        target_lines = tgt_file.readlines()
    res = []
    for i in range(len(target_lines)):
        if i % 5 == 0 and target_lines[i].strip() not in res:
            res.append(target_lines[i].strip())
    target_lines = res

    return target_lines


def load_prompt(source_file, target_file, sub_size=None):
    target_lines = open(target_file, encoding="utf-8").read().splitlines()
    source_lines = open(source_file, encoding="utf-8").read().splitlines()

    if sub_size is not None and sub_size < len(source_lines):
        indeces = np.random.choice(len(source_lines), sub_size, replace=False)
        source_lines = [source_lines[i] for i in indeces]
        target_lines = [target_lines[i] for i in indeces]

    source = []
    target = []
    for i in range(len(source_lines)):
        source_text = source_lines[i]
        target_text = target_lines[i]
        prefix_text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction:{source_text} ### Response:"
        source.append(prefix_text)
        target.append(target_text)

    return source, target


def load_response(source_file, target_file):
    target_lines = open(target_file, encoding="utf-8").read().splitlines()
    source_lines = open(source_file, encoding="utf-8").read().splitlines()

    res = []
    for i in range(len(source_lines)):
        prefix_text = f"Response:{target_lines[i]}</s>"
        if prefix_text not in res:
            res.append(prefix_text)

    return res


class Trie(object):
    def __init__(self, sequences: List[List[int]] = []):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1

        self.append_trie = None
        self.bos_token_id = None

    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        return Trie._get_from_trie(
            prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id
        )

    @staticmethod
    def load_from_dict(trie_dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
            prefix_sequence: List[int],
            trie_dict: Dict,
            append_trie=None,
            bos_token_id: int = None,
    ):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return []

    def __iter__(self):
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(
                        prefix_sequence + [next_token], trie_dict[next_token]
                    )
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)


class T5Dataset(Dataset):
    def __init__(
            self,
            tokenizer,
            source_file,
            target_file,
            max_source_len=128,
            max_target_len=8,
            add_prefix=False,
            subset_size=None,
    ):
        self.tokenizer = tokenizer
        self.source_texts = open(
            source_file, encoding="utf-8").read().splitlines()
        self.target_texts = open(
            target_file, encoding="utf-8").read().splitlines()
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.subset_size = subset_size
        self.add_prefix = add_prefix

        if self.subset_size is not None:
            indices = list(range(len(self.source_texts)))
            sampled_indices = random.sample(indices, self.subset_size)
            self.source_texts = [self.source_texts[i] for i in sampled_indices]
            self.target_texts = [self.target_texts[i] for i in sampled_indices]

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        source_text = self.source_texts[idx]
        if self.add_prefix:
            source_text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. Instruction:{source_text} Response:"
        target_text = self.target_texts[idx]

        source_encodings = self.tokenizer(
            source_text,
            padding="max_length",
            max_length=self.max_source_len,
            truncation=True,
            return_tensors="pt",
        )
        target_encodings = self.tokenizer(
            target_text,
            padding="max_length",
            max_length=self.max_target_len,
            truncation=True,
            return_tensors="pt",
        )
        # print(target_encodings)

        input_ids = source_encodings["input_ids"]
        labels = target_encodings["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {"input_ids": input_ids.squeeze(), "labels": labels.squeeze()}
        # return  {"input_ids": source_text, "labels": target_text}


class LLaMaDataset(Dataset):
    def __init__(self, tokenizer, source_file, target_file, subset_size=None):
        self.tokenizer = tokenizer
        self.source_texts = open(
            source_file, encoding="utf-8").read().splitlines()
        self.target_texts = open(
            target_file, encoding="utf-8").read().splitlines()
        self.subset_size = subset_size

        if self.subset_size is not None:
            indices = list(range(len(self.source_texts)))
            sampled_indices = random.sample(indices, self.subset_size)
            self.source_texts = [self.source_texts[i] for i in sampled_indices]
            self.target_texts = [self.target_texts[i] for i in sampled_indices]

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        source_text = self.source_texts[idx]
        target_text = self.target_texts[idx]
        return preprocess_function(source_text, target_text, self.tokenizer)


class T5ForPAGSeqIdGeneration(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)
        if not hasattr(config, 'pag_code_book_size'):
            raise ValueError(
                "Config must have 'pag_code_book_size' attribute for T5ForPAGSeqIdGeneration. This should be set to the number of 'c_...' tokens.")
        self.pag_code_book_size = config.pag_code_book_size
        self.seq_id_preference_head = nn.Linear(config.d_model, self.pag_code_book_size)

        if hasattr(config, 'c_token_start_id') and config.c_token_start_id is not None:
            with torch.no_grad():
                lm_head_weights = self.lm_head.weight  # shape: [vocab_size, d_model]
                c_token_start = config.c_token_start_id
                if c_token_start < lm_head_weights.shape[0] and c_token_start + self.pag_code_book_size <= \
                        lm_head_weights.shape[0]:
                    c_token_weights = lm_head_weights[c_token_start:c_token_start + self.pag_code_book_size, :]
                    self.seq_id_preference_head.weight.data = c_token_weights
                    if self.seq_id_preference_head.bias is not None:
                        self.seq_id_preference_head.bias.data.zero_()

                    print(
                        f"Initialized seq_id_preference_head with lm_head weights for c_tokens [{c_token_start}:{c_token_start + self.pag_code_book_size}]")
                else:
                    print(
                        f"Warning: c_token_start_id {c_token_start} is out of range for lm_head. Using random initialization for seq_id_preference_head.")
        else:
            print(
                "Warning: config does not have c_token_start_id. Using random initialization for seq_id_preference_head.")

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,  # Ensure encoder_last_hidden_state is returned
            return_dict=return_dict,
        )

        h_q_seq = None

        # Determine the source of encoder_last_hidden_state
        current_encoder_last_hidden_state = None
        if return_dict and hasattr(outputs,
                                   'encoder_last_hidden_state') and outputs.encoder_last_hidden_state is not None:
            current_encoder_last_hidden_state = outputs.encoder_last_hidden_state
        elif encoder_outputs is not None:
            if isinstance(encoder_outputs, tuple):
                current_encoder_last_hidden_state = encoder_outputs[0]
            else:  # Assuming it's a BaseModelOutput-like object
                current_encoder_last_hidden_state = encoder_outputs.last_hidden_state

        if current_encoder_last_hidden_state is not None:
            # [batch_size, seq_len, d_model] -> [batch_size, d_model]
            pooled_encoder_output = torch.max(current_encoder_last_hidden_state, dim=1)[0]
            preference_logits = self.seq_id_preference_head(pooled_encoder_output)  # [batch_size, pag_code_book_size]
            h_q_seq = torch.log1p(torch.relu(preference_logits))

        if return_dict:
            if isinstance(outputs, Seq2SeqLMOutput):
                outputs.h_q_seq = h_q_seq
                return outputs
            elif isinstance(outputs, dict):
                outputs['h_q_seq'] = h_q_seq
                return outputs
        else:
            return outputs, h_q_seq


class PAGLogitsProcessor(LogitsProcessor):
    def __init__(self, h_q_seq_batch_expanded: torch.Tensor, lambda_guidance: float, c_token_start_id: int,
                 code_book_size: int):
        self.h_q_seq_batch_expanded = h_q_seq_batch_expanded.to(torch.float32)  # Ensure dtype
        self.lambda_guidance = lambda_guidance
        self.c_token_start_id = c_token_start_id
        self.code_book_size = code_book_size

    def __call__(self, input_ids, scores):
        step = input_ids.shape[1]
        if step == 1:
            return scores

        # raw scores for the first c_ token of each hypo
        raw_first_c = scores[:, self.c_token_start_id].clone()

        bias = torch.zeros_like(scores)
        bias[:, self.c_token_start_id:self.c_token_start_id + self.code_book_size] += \
            self.lambda_guidance * self.h_q_seq_batch_expanded
        new_scores = scores + bias

        # new scores after guidance
        guided_first_c = new_scores[:, self.c_token_start_id]

        # if torch.distributed.get_rank() == 0 and step == 2:
        #     print("raw vs guided (first 5 hypos):")
        #     for r, g in zip(raw_first_c[:5], guided_first_c[:5]):
        #         print(f"{r.item():.4f} -> {g.item():.4f}")
        #

        return new_scores

class PrefixPriorLogitsProcessor(LogitsProcessor):
    """
    
      (i) token-level  λ·h_q[v]
      (ii) prefix-level  max_global(prefix)
    """
    def __init__(self,
                 h_q_expanded,          # [B*beam, C]
                 λ,
                 c_start, code_book_size,
                 prefix_priors,         # List[Dict[prefix_tuple -> float]]，length = B
                 beam_size):
        self.h_q_expanded  = h_q_expanded
        self.λ             = λ
        self.c_start       = c_start
        self.C             = code_book_size
        self.prefix_priors = prefix_priors
        self.beam_size     = beam_size

    def __call__(self, input_ids, scores):
        # -------- token-level  --------
        bias = torch.zeros_like(scores)
        bias[:, self.c_start:self.c_start+self.C] += self.λ * self.h_q_expanded
        scores = scores + bias
        # -------- prefix-level  --------
        for i in range(scores.size(0)):
            q_idx  = i // self.beam_size
            prefix = tuple(input_ids[i].tolist())
            prior  = self.prefix_priors[q_idx].get(prefix, 0)
            scores[i] += prior
        return scores

class QueryEvalCallback(TrainerCallback):
    def __init__(
            self,
            local_rank,
            test_dataset_1,
            test_dataset_2,
            tgt_file,
            logger,
            batch_size,
            collator,
            tokenizer,
            wandb=None,
            log_freq=3,
            gen_len=20,
            lambda_guidance=0.0,
            c_token_start_id=None,
            code_book_size=None
    ):
        self.tokenizer = tokenizer
        self.logger = logger
        self.test_dataset_1 = test_dataset_1
        self.test_dataset_2 = test_dataset_2
        self.dataloader_1 = DataLoader(
            test_dataset_1,
            batch_size=32,
            collate_fn=collator,
            shuffle=True,
            drop_last=False,
            num_workers=10,
        )
        self.dataloader_2 = DataLoader(
            test_dataset_2,
            batch_size=32,
            collate_fn=collator,
            shuffle=True,
            drop_last=False,
            num_workers=10,
        )
        self.code_list = load_codes(tgt_file)

        # ---------- test.target →  seq_ids_tensor ----------
        seq_lists = []
        with open(tgt_file, encoding="utf-8") as fh:   # tgt_file = test_target_file
            for line in fh:
                seq_lists.append([int(tok.lstrip("c_")) for tok in line.strip().split()])

        self.seq_len = max(len(x) for x in seq_lists)  # m
        for row in seq_lists:                          #
            row += [0]*(self.seq_len - len(row))

        # [N_img, m]
        self.seq_ids_tensor = torch.tensor(seq_lists, dtype=torch.long)
        self.n_candidates   = 500      # Top-n


        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.candidate_trie = Trie([[pad_token_id] + self.tokenizer.encode(x) for x in self.code_list])
        self.test_prefix_allowed_tokens_fn = prefix_allowed_tokens_fn(self.candidate_trie)
        self.wandb = wandb if wandb else None
        self.log_freq = log_freq
        self.gen_len = gen_len
        self.local_rank = local_rank
        self.lambda_guidance = lambda_guidance
        self.c_token_start_id = c_token_start_id
        self.code_book_size = code_book_size

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.local_rank != 0:
            return
        current_epoch = state.epoch
        if not (state.epoch == int(state.epoch) and int(state.epoch) % self.log_freq == 0 and int(state.epoch) > 0):
            if not (int(state.epoch) == args.num_train_epochs and args.num_train_epochs % self.log_freq != 0):
                return
        warmup_epochs = 8
        lambda_max = 1 # coco-0.1# flick-1
        tau_high = 12.0
        tau_low  = 8.0
        if current_epoch < warmup_epochs:
            cur_lambda = lambda_max * (current_epoch / warmup_epochs)
            cur_tau    = tau_high - (tau_high - tau_low) * (current_epoch / warmup_epochs)
        else:
            cur_lambda = lambda_max
            cur_tau    = tau_low
        recall_count_at_1_1 = 0
        recall_count_at_5_1 = 0
        recall_count_at_10_1 = 0
        recall_count_at_1_2 = 0
        recall_count_at_5_2 = 0
        recall_count_at_10_2 = 0

        model = kwargs["model"]
        model.eval()

        num_beams_eval = 10
        generation_config_eval = GenerationConfig(
            num_beams=num_beams_eval,
            max_new_tokens=self.gen_len,
            num_return_sequences=num_beams_eval,
            early_stopping=True,
            use_cache=True,
        )

        for batch_1 in tqdm(self.dataloader_1, desc="Evaluating subset of train queries (global guided)"):
            inputs_1 = batch_1
            input_ids_1 = inputs_1["input_ids"].to(model.device)
            attention_mask_1 = inputs_1["attention_mask"].to(model.device)

            logits_processor_list_1 = []
            if self.lambda_guidance > 0 and self.c_token_start_id is not None and self.code_book_size is not None:
                with torch.no_grad():
                    # Prepare minimal decoder_input_ids for the planning pass
                    # T5 uses config.decoder_start_token_id (usually pad_token_id) to start decoding
                    decoder_start_token_id = model.config.decoder_start_token_id
                    if decoder_start_token_id is None:  # Fallback if not set, though it should be
                        decoder_start_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

                    batch_size_1 = input_ids_1.shape[0]
                    dummy_decoder_input_ids_1 = torch.full(
                        (batch_size_1, 1),
                        decoder_start_token_id,
                        dtype=torch.long,
                        device=model.device
                    )

                    planning_outputs_1 = model(
                        input_ids=input_ids_1,
                        attention_mask=attention_mask_1,
                        decoder_input_ids=dummy_decoder_input_ids_1,  # Provide dummy decoder input
                        return_dict=True
                        # No labels needed here as we only want encoder_outputs for h_q_seq
                    )
                if hasattr(planning_outputs_1, 'h_q_seq') and planning_outputs_1.h_q_seq is not None:
                    h_q_seq_batch_original_1 = planning_outputs_1.h_q_seq
                    h_q_seq_batch_expanded_1 = h_q_seq_batch_original_1.repeat_interleave(num_beams_eval, dim=0)
                    pag_processor_1 = PAGLogitsProcessor(
                        h_q_seq_batch_expanded_1, self.lambda_guidance, self.c_token_start_id, self.code_book_size
                    )
                    logits_processor_list_1.append(pag_processor_1)
                else:
                    self.logger.warning("h_q_seq not found in model_1 output during PAG eval on train subset.")

            with torch.no_grad():
                batch_beams_1 = model.generate(
                    input_ids_1,
                    attention_mask=attention_mask_1,
                    generation_config=generation_config_eval,
                    logits_processor=logits_processor_list_1 if logits_processor_list_1 else None,
                ).reshape(input_ids_1.shape[0], num_beams_eval, -1)

            for beams, label_ids in zip(batch_beams_1, inputs_1["labels"]):
                rank_list_strings = self.tokenizer.batch_decode(beams, skip_special_tokens=True)
                rank_list_of_lists = [x.split(" ") for x in rank_list_strings]

                label_ids[label_ids == -100] = self.tokenizer.pad_token_id
                label_string = self.tokenizer.decode(label_ids, skip_special_tokens=True).strip()
                label_list = label_string.split(" ")

                hits = [i for i, x_list in enumerate(rank_list_of_lists) if x_list == label_list]
                if hits:
                    if hits[0] < 10:
                        recall_count_at_10_1 += 1
                        if hits[0] < 5:
                            recall_count_at_5_1 += 1
                            if hits[0] == 0:
                                recall_count_at_1_1 += 1

        hits_at_1_data_1 = recall_count_at_1_1 / len(self.test_dataset_1)
        hits_at_5_data_1 = recall_count_at_5_1 / len(self.test_dataset_1)
        hits_at_10_data_1 = recall_count_at_10_1 / len(self.test_dataset_1)

        log_msg_1 = f"Epoch {state.epoch:.2f} training set subset: Recall@1: {hits_at_1_data_1:.4f}, Recall@5: {hits_at_5_data_1:.4f}, Recall@10: {hits_at_10_data_1:.4f}"
        self.logger.info(log_msg_1)
        if self.wandb and self.local_rank == 0:
            self.wandb.log({
                " Train Recall@1": hits_at_1_data_1,
                " Train Recall@5": hits_at_5_data_1,
                " Train Recall@10": hits_at_10_data_1,
            })

        for batch_2 in tqdm(self.dataloader_2, desc="Evaluating test queries (global guided)"):
            inputs_2 = batch_2
            input_ids_2 = inputs_2["input_ids"].to(model.device)
            attention_mask_2 = inputs_2["attention_mask"].to(model.device)

            logits_processor_list_2 = []
            if self.lambda_guidance > 0 and self.c_token_start_id is not None and self.code_book_size is not None:
                with torch.no_grad():
                    # Prepare minimal decoder_input_ids for the planning pass
                    decoder_start_token_id = model.config.decoder_start_token_id
                    if decoder_start_token_id is None:
                        decoder_start_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

                    batch_size_2 = input_ids_2.shape[0]
                    dummy_decoder_input_ids_2 = torch.full(
                        (batch_size_2, 1),
                        decoder_start_token_id,
                        dtype=torch.long,
                        device=model.device
                    )
                    planning_outputs_2 = model(
                        input_ids=input_ids_2,
                        attention_mask=attention_mask_2,
                        decoder_input_ids=dummy_decoder_input_ids_2,  # Provide dummy decoder input
                        return_dict=True
                    )
                if hasattr(planning_outputs_2, 'h_q_seq') and planning_outputs_2.h_q_seq is not None:
                    h_q_seq_batch_original_2 = planning_outputs_2.h_q_seq
                    # ---------- prefix-prior（ ----------
                    seq_ids_gpu = self.seq_ids_tensor.to(model.device)  # [N_img, m]
                    B_cur, C = h_q_seq_batch_original_2.shape
                    N_img, m = seq_ids_gpu.shape

                    # (a) s_global = sum h_q over SeqID
                    scores_tok = (
                        h_q_seq_batch_original_2.unsqueeze(1).expand(-1, N_img, -1)
                        .gather(2, seq_ids_gpu.unsqueeze(0).expand(B_cur, -1, -1))
                    )  # [B, N_img, m]
                    s_global = scores_tok.sum(-1)  # [B, N_img]

                    # (b) Top-n & prefix→max
                    top_vals, top_idx = torch.topk(s_global, k=self.n_candidates, dim=-1)
                    batch_prefix_priors = []
                    offset = self.c_token_start_id
                    for b in range(B_cur):
                        T = {}
                        for j, img_ix in enumerate(top_idx[b].tolist()):
                            raw = top_vals[b, j].item()  # g_score_raw
                            seq = seq_ids_gpu[img_ix].tolist()
                            m_eff = len([x for x in seq if x != 0])
                            # ★ 用 cur_tau
                            g_score = raw / (m_eff * cur_tau)

                            for t in range(1, m_eff + 1):
                                bos_id = model.config.decoder_start_token_id
                                pref = (bos_id,) + tuple(x + offset for x in seq[:t])
                                #
                                T[pref] = max(T.get(pref, 0.0), g_score)
                        batch_prefix_priors.append(T)

                    # (c) beam*B
                    h_q_seq_expanded_2 = h_q_seq_batch_original_2.repeat_interleave(num_beams_eval, dim=0)

                    # (d) logits-processor
                    prefix_proc = PrefixPriorLogitsProcessor(
                        h_q_expanded=h_q_seq_expanded_2,
                        λ=cur_lambda,
                        c_start=self.c_token_start_id,
                        code_book_size=self.code_book_size,
                        prefix_priors=batch_prefix_priors,
                        beam_size=num_beams_eval
                    )
                    logits_processor_list_2.append(prefix_proc)
                else:
                    self.logger.warning("h_q_seq not found in model output during PAG eval on test set.")

            with torch.no_grad():
                batch_beams_2 = model.generate(
                    input_ids_2,
                    attention_mask=attention_mask_2,
                    generation_config=generation_config_eval,
                    logits_processor=logits_processor_list_2 if logits_processor_list_2 else None,
                    prefix_allowed_tokens_fn=self.test_prefix_allowed_tokens_fn,
                ).reshape(input_ids_2.shape[0], num_beams_eval, -1)

            for beams, label_ids in zip(batch_beams_2, inputs_2["labels"]):
                rank_list_strings = self.tokenizer.batch_decode(beams, skip_special_tokens=True)
                rank_list_of_lists = [x.split(" ") for x in rank_list_strings]

                label_ids[label_ids == -100] = self.tokenizer.pad_token_id
                label_string = self.tokenizer.decode(label_ids, skip_special_tokens=True).strip()
                label_list = label_string.split(" ")

                hits = [i for i, x_list in enumerate(rank_list_of_lists) if x_list == label_list]
                if hits:
                    if hits[0] < 10:
                        recall_count_at_10_2 += 1
                        if hits[0] < 5:
                            recall_count_at_5_2 += 1
                            if hits[0] == 0:
                                recall_count_at_1_2 += 1

        if len(self.test_dataset_1) > 0:
            hits_at_1_data_2 = recall_count_at_1_2 / len(self.test_dataset_2)
            hits_at_5_data_2 = recall_count_at_5_2 / len(self.test_dataset_2)
            hits_at_10_data_2 = recall_count_at_10_2 / len(self.test_dataset_2)
        else:
            hits_at_1_data_2 = hits_at_5_data_2 = hits_at_10_data_2 = 0.0

        log_msg_2 = f"Epoch {state.epoch:.2f} test set: Recall@1: {hits_at_1_data_2:.4f}, Recall@5: {hits_at_5_data_2:.4f}, Recall@10: {hits_at_10_data_2:.4f}\n"
        self.logger.info(log_msg_2)
        if self.wandb and self.local_rank == 0:
            self.wandb.log({
                " Test Recall@1": hits_at_1_data_2,
                " Test Recall@5": hits_at_5_data_2,
                " Test Recall@10": hits_at_10_data_2,
            })

        model.train()


class TrainerwithTemperature(Trainer):
    def __init__(self, temperature=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        logits = logits / self.temperature
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

