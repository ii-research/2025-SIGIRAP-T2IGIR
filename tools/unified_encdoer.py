import argparse
import torch
import os
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch.nn as nn

class UnifiedTIRModel(T5ForConditionalGeneration):
    def __init__(self, config, code_book_size=1024):
        super().__init__(config)
        self.seq_id_preference_head = nn.Linear(config.d_model, code_book_size)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_dir', required=True)
    parser.add_argument('--encoder_ckpt', required=True)
    parser.add_argument('--guidance_head_ckpt', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--code_book_size', type=int, default=1024)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # 1. 加载
    print(f"Loading base model from: {args.base_model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir)
    model = UnifiedTIRModel.from_pretrained(args.base_model_dir, code_book_size=args.code_book_size)

    # 2. Encoder 融合
    print(f"Loading encoder from: {args.encoder_ckpt}")
    if os.path.exists(args.encoder_ckpt):
        enc_sd = torch.load(args.encoder_ckpt, map_location='cpu')
        filtered = {k.replace("encoder.", "", 1): v for k,v in enc_sd.items() if k.startswith("encoder.") and not k.startswith("encoder.embed_tokens")}
        model.encoder.load_state_dict(filtered, strict=False)
    else:
        print(f"Warning: Encoder checkpoint not found.")

    # 3. Guidance head 融合
    print(f"Loading seq_id_preference_head from: {args.guidance_head_ckpt}")
    if os.path.exists(args.guidance_head_ckpt):
        head_sd = torch.load(args.guidance_head_ckpt, map_location='cpu')
        try:
            model.seq_id_preference_head.load_state_dict(head_sd, strict=True)
        except Exception as e:
            print(f"Loading guidance head failed: {e}. Try clean keys.")
            head_sd = {k.replace("module.", ""): v for k,v in head_sd.items()}
            model.seq_id_preference_head.load_state_dict(head_sd, strict=True)
    else:
        print(f"Warning: Guidance head checkpoint not found.")

    # 4. 保存
    print(f"Saving merged model to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")
