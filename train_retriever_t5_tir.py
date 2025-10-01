import os
from torch.utils.data import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from transformers import Trainer, TrainingArguments, TrainerCallback, DataCollatorWithPadding
import torch
import logging
import wandb
import argparse
from utils_tir import QueryEvalCallback, TrainerwithTemperature, T5Dataset, T5ForPAGSeqIdGeneration
from peft import TaskType, LoraConfig, get_peft_model, PeftModel
from datetime import datetime



def parse_args():
    parser = argparse.ArgumentParser(description="Train T5 model with RQ-VAE codes")

    parser.add_argument('--data_path', type=str, default='data/flickr/flickr_codes', help='data path')
    parser.add_argument('--output_dir', type=str, default='output/flickr', help='output directory')
    parser.add_argument('--model_name', type=str, default='t5-base', help='model name')
    parser.add_argument('--train_epoch', type=int, default=100, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--train_batch_size', type=int, default=128, help='training batch size')
    parser.add_argument('--code_book_size', type=int, default=1024, help='code book size for general case, or for c_ tokens if code_book_num==1')
    parser.add_argument('--code_book_num', type=int, default=1, help='number of code books')
    parser.add_argument('--add_embedding', action='store_true', help='add rq_embedding to tokenizer')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--log_freq', type=int, default=5, help='eval log frequency')
    parser.add_argument('--source_length', type=int, default=128, help='source length')
    parser.add_argument('--target_length', type=int, default=8, help='target length')
    parser.add_argument('--gen_len', type=int, default=20, help='generation length')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup ratio')
    parser.add_argument('--eval_strategy', type=str, default='epoch', help='evaluation strategy')
    parser.add_argument('--save_strategy', type=str, default='epoch', help='save strategy')
    parser.add_argument('--save_total_limit', type=int, default=5, help='save total limit')
    parser.add_argument('--logging_steps', type=int, default=100, help='logging steps')
    parser.add_argument('--deepseed_config', type=str, default=None, help='deepspeed config file')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank')
    parser.add_argument('--temperature', type=float, default=1.0, help='softmax temperature')
    parser.add_argument('--add_prefix', action='store_true', help='add prefix to inputs')
    parser.add_argument('--float16', action='store_true', help='use float16')
    parser.add_argument('--bf16', action='store_true', help='use bf16')
    parser.add_argument('--lora', action='store_true', help='use lora')
    parser.add_argument('--lambda_guidance', type=float, default=0.5, help='Lambda for TIR guidance strength during generation.')
    parser.add_argument('--pretrained_head_path', type=str, default=None,
                        help='Path to pretrained seq_id_preference_head weights')

    return parser.parse_args()


if __name__ == '__main__':

    train_args = parse_args()
    data_path = train_args.data_path

    print('training on: ', data_path)

    model_name = train_args.model_name
    train_source_file = data_path + '/train.source'
    train_target_file = data_path + '/train.target'
    val_source_file = data_path + '/val.source'
    val_target_file = data_path + '/val.target'
    test_source_file = data_path + '/test.source'
    test_target_file = data_path + '/test.target'

    train_epoch = train_args.train_epoch
    learning_rate = train_args.learning_rate
    train_batch_size = train_args.train_batch_size
    # General code_book_size from arguments
    general_code_book_size_arg = train_args.code_book_size
    code_book_num = train_args.code_book_num
    add_embedding = train_args.add_embedding
    dropout_rate = train_args.dropout_rate
    log_freq = train_args.log_freq
    source_length = train_args.source_length
    target_length = train_args.target_length
    gen_len = train_args.gen_len

    # --- num_c_tokens CALCULATION ---
    num_c_tokens = 0
    if code_book_num == 1:
        # When code_book_num is 1, general_code_book_size_arg is assumed to be for 'c_' tokens
        num_c_tokens = general_code_book_size_arg
    else:
        # Logic for 'c_' tokens if they exist in multi-codebook scenario would be needed here.
        # For now, assuming 'c_' tokens for PAG are only used when code_book_num == 1.
        print("Warning: code_book_num != 1. PAG guidance for 'c_' tokens might be misconfigured or ineffective if 'c_' tokens are not specifically handled.")

    if num_c_tokens == 0 and train_args.lambda_guidance > 0:
        print(f"Warning: num_c_tokens is 0 (code_book_num={code_book_num}, code_book_size={general_code_book_size_arg}) but lambda_guidance is {train_args.lambda_guidance}. PAG guidance will be ineffective.")
    # --- END num_c_tokens CALCULATION ---

    rq_signal = 'embadded' if add_embedding else 'noemb'
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    # Use general_code_book_size_arg for consistent output directory naming as before
    output_dir = current_time+'_'+str(data_path.split('/')[-1])+'_c'+str(general_code_book_size_arg)+'_ep' + \
        str(train_epoch)+'_lr'+str(learning_rate)+'_bch'+str(train_batch_size)+'_' + rq_signal

    local_rank = int(os.environ.get("LOCAL_RANK") or 0)

    if local_rank == 0:
        wandb.login()
        wandb.init(project='AVG_retriever', name=output_dir)

    output_dir_name = train_args.output_dir + '/' + train_args.model_name.split('/')[-1] + '/' + output_dir

    # Load config and add pag_code_book_size
    config = T5Config.from_pretrained(model_name)
    config.dropout_rate = dropout_rate
    config.pag_code_book_size = num_c_tokens if num_c_tokens > 0 else 1 # For T5ForPAGSeqIdGeneration, avoid 0

    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # Define prefix for multi-codebook scenario
    prefix_list = ['a_', 'b_', 'c_', 'd_'] # Renamed from 'prefix' to 'prefix_list'

    extra_tokens = []
    if code_book_num == 1: # This implies we are using 'c_' tokens for PAG
        # Here, general_code_book_size_arg correctly refers to num_c_tokens
        for count in range(general_code_book_size_arg):
            extra_tokens.append('c_'+str(count))
    else: # Logic for other types of tokens if code_book_num != 1
        for cb_idx in range(code_book_num):
            current_prefix = prefix_list[cb_idx % len(prefix_list)]
            # general_code_book_size_arg is for each of these other codebooks
            for count in range(general_code_book_size_arg):
                extra_tokens.append(current_prefix + str(count))
    if local_rank == 0:
        print('Number of extra tokens to add:', len(extra_tokens))
    if extra_tokens:
        tokenizer.add_tokens(extra_tokens)
        if local_rank == 0:
            print(f"Added {len(extra_tokens)} tokens. New tokenizer_len: {len(tokenizer)}")
    else:
        print("No extra tokens were specified to be added.")

    # --- c_token_start_id CALCULATION ---
    c_token_start_id = None
    if num_c_tokens > 0:
        try:
            first_c_token_str = 'c_0'
            if first_c_token_str in tokenizer.get_vocab(): # Check if 'c_0' is known after add_tokens
                 c_token_start_id = tokenizer.convert_tokens_to_ids(first_c_token_str)
                 if c_token_start_id == tokenizer.unk_token_id:
                     print(f"Warning: Token '{first_c_token_str}' resolved to UNK token. PAG guidance might not work correctly.")
                     c_token_start_id = None
                 else:
                    # Sanity check (optional but good)
                    if num_c_tokens > 1: # Only check last if more than one c_token
                        last_c_token_str = f'c_{num_c_tokens-1}'
                        expected_last_c_id = c_token_start_id + num_c_tokens - 1
                        actual_last_c_id = tokenizer.convert_tokens_to_ids(last_c_token_str)
                        if actual_last_c_id != expected_last_c_id:
                            print(f"Warning: 'c_...' tokens may not be contiguous after adding. Start ID for '{first_c_token_str}': {c_token_start_id}, Expected Last ID for {last_c_token_str}: {expected_last_c_id}, Actual Last ID: {actual_last_c_id}")
            else:
                print(f"Warning: num_c_tokens > 0 but token '{first_c_token_str}' was not found in tokenizer vocab after adding tokens. PAG guidance will be ineffective.")
                c_token_start_id = None
        except Exception as e:
            print(f"Warning: Could not get ID for 'c_0'. PAG guidance might not work correctly. Error: {e}")
            c_token_start_id = None
    # --- END c_token_start_id CALCULATION ---

    if c_token_start_id is not None:
        config.c_token_start_id = c_token_start_id

    # Load model AFTER config is modified and BEFORE tokenizer is resized with model
    if train_args.float16:
        torch_dtype = torch.float16
    elif train_args.bf16:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    model = T5ForPAGSeqIdGeneration.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch_dtype
    )



    model.resize_token_embeddings(len(tokenizer))

    # Load pretrained seq_id_preference_head if specified
    pretrained_head_path = train_args.pretrained_head_path
    if os.path.exists(pretrained_head_path):
        state_dict = torch.load(pretrained_head_path, map_location='cpu')
        model.seq_id_preference_head.load_state_dict(state_dict)
        if local_rank == 0:
            print("Load pretrained seq_id_preference_head if specified  ")
    else:
        print(f"no：{pretrained_head_path}， skipping loading pretrained seq_id_preference_head.")

    if add_embedding:
        rq_emb = torch.load(data_path+'/codebook_embedding.pt')
        token_embeddings = model.get_input_embeddings()
        num_newly_added_tokens = len(extra_tokens)
        idx_of_first_new_token = len(tokenizer) - num_newly_added_tokens

        if rq_emb.size(0) == num_newly_added_tokens:
             token_embeddings.weight.data[idx_of_first_new_token : idx_of_first_new_token + num_newly_added_tokens] = rq_emb
             if local_rank == 0:
                print(f'RQ codebook_embedding (size {rq_emb.size(0)}) added for the {num_newly_added_tokens} extra tokens.')
        else:
            print(f"Warning: rq_emb size {rq_emb.size(0)} does not match number of extra tokens {num_newly_added_tokens}. Skipping direct embedding loading.")

    if train_args.lora:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            modules_to_save=['embed_tokens', 'lm_head', 'seq_id_preference_head'], # Added seq_id_preference_head
            lora_dropout=0.1,
            bias="none",
            inference_mode=False,
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    reporter = ['wandb'] if local_rank == 0 else "none"
    training_args = TrainingArguments(
        output_dir=output_dir_name,

        num_train_epochs=train_epoch,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        dataloader_num_workers=10,

        warmup_ratio=train_args.warmup_ratio,
        learning_rate=learning_rate,

        logging_dir=output_dir_name+'/logs/',
        report_to=reporter,
        evaluation_strategy=train_args.eval_strategy,

        save_strategy=train_args.save_strategy,
        save_total_limit=train_args.save_total_limit,

        logging_steps=train_args.logging_steps,

        deepspeed=train_args.deepseed_config,
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,

        metric_for_best_model="eval_loss", # Or a relevant recall metric from callback
        save_only_model=True,

        fp16=train_args.float16,
        bf16=train_args.bf16,
    )
    model.config.use_cache = False

    train_dataset = T5Dataset(tokenizer, train_source_file, train_target_file, max_source_len=source_length, max_target_len=target_length,
                              add_prefix=train_args.add_prefix)
    val_dataset = T5Dataset(tokenizer, val_source_file, val_target_file, max_source_len=source_length, max_target_len=target_length,
                            add_prefix=train_args.add_prefix)
    test_dataset = T5Dataset(tokenizer, test_source_file, test_target_file, max_source_len=source_length, max_target_len=target_length,
                             add_prefix=train_args.add_prefix)
    sub_train_dataset = T5Dataset(tokenizer, train_source_file, train_target_file, max_source_len=source_length, max_target_len=target_length,
                                  add_prefix=train_args.add_prefix, subset_size=1000)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='max_length', max_length=source_length)

    os.makedirs(output_dir_name, exist_ok=True)
    logging.basicConfig(filename=output_dir_name+'/training_log.log', level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger(__name__)

    if local_rank == 0:
        logger.info('training arguments: '+str(train_args))
        logger.info('training dataset size: '+str(len(train_dataset)))
        logger.info('validation dataset size: '+str(len(val_dataset)))
        logger.info('test dataset size: '+str(len(test_dataset)))
        logger.info('transformers training_args: '+str(training_args))

    class MultiTaskTrainer(Trainer):
        def __init__(self, lambda_guidance=0.3, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.lambda_guidance = lambda_guidance

        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            h_q_seq = outputs.h_q_seq  # [batch, codebook_size]
            #
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            ce_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            # InfoNCE
            batch_size = labels.size(0)
            # 
            positive_token_ids = labels[:, :4]  # [batch, 4]#
            c_token_start_id = 32100
            codebook_size = 1024 # Assuming codebook_size is 1024 for general case
            positive_code_ids = positive_token_ids - c_token_start_id  # [batch, 4]
            # if positive_code_ids.size(1) != 4:
            if (positive_code_ids < 0).any() or (positive_code_ids >= codebook_size).any():
                print("code id:", positive_code_ids)
                raise ValueError("Some code ids are out of codebook range!")

            log_probs = torch.log_softmax(h_q_seq, dim=1)
            pos_log_probs = log_probs.gather(1, positive_code_ids)
            info_nce_loss = -pos_log_probs.mean()

            total_loss = ce_loss + self.lambda_guidance * info_nce_loss

            return (total_loss, outputs) if return_outputs else total_loss

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[QueryEvalCallback(local_rank=local_rank,
                                     test_dataset_1=sub_train_dataset,
                                     test_dataset_2=val_dataset,
                                     tgt_file=val_target_file,
                                     logger=logger,
                                     batch_size=train_args.train_batch_size,
                                     collator=data_collator,
                                     tokenizer=tokenizer,
                                     wandb=wandb if local_rank == 0 else None,
                                     log_freq=log_freq,
                                     gen_len=gen_len,
                                     lambda_guidance=train_args.lambda_guidance,
                                     c_token_start_id=c_token_start_id,
                                     code_book_size=num_c_tokens if num_c_tokens > 0 else 0)],
        lambda_guidance=train_args.lambda_guidance,
    )
    # -------------- sanity‑check  --------------
    if local_rank == 0:
        print("λ_guidance =", train_args.lambda_guidance)
        print("num_c_tokens =", num_c_tokens)
        print("c_token_start_id =", c_token_start_id)

    if train_args.lambda_guidance > 0:
        assert num_c_tokens > 0, " c_ tokens is wrong"
        assert c_token_start_id is not None, "no ID"

        last_id_expect = c_token_start_id + num_c_tokens - 1
        last_id_actual = tokenizer.convert_tokens_to_ids(f"c_{num_c_tokens-1}")
        if local_rank == 0:
            print("Expect c_token range:", c_token_start_id, "–", last_id_expect)
            print("Actual last c_token id:", last_id_actual)
        assert last_id_actual == last_id_expect, "c_ tokens is wrong！"
    if local_rank == 0:
        print("=== sanity check passed ===\n")
    # -------------- sanity‑check end ----------------


    trainer.train()
