import os, sys, json, numpy as np, torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

BLIP_ROOT = "/home/xxx/blip"
sys.path.insert(0, BLIP_ROOT)

BATCH_IMG = 64
BATCH_TXT = 256
calc_recall = lambda r, k: float((r < k).mean())      #

# ---------- helper: image CLS ----------
def encode_images(model, tfm, img_dir, names, device):
    feats = []
    for i in tqdm(range(0, len(names), BATCH_IMG), desc="Image enc"):
        batch = names[i:i+BATCH_IMG]
        imgs = [tfm(Image.open(os.path.join(img_dir, n)).convert("RGB")) for n in batch]
        imgs = torch.stack(imgs).to(device)
        with torch.no_grad():
            f = model.visual_encoder(imgs)[:, 0]
            if hasattr(model, "vision_proj"):
                f = model.vision_proj(f)
        feats.append(f.cpu())
    return torch.nn.functional.normalize(torch.cat(feats, 0), dim=-1)

# ---------- helper: text CLS ----------
def encode_texts(model, caps, device):
    feats = []
    with torch.no_grad():
        dummy_dim = model.visual_encoder(torch.zeros(1,3,384,384,device=device))[:,0].shape[-1]
    dummy_img  = torch.zeros((1,1,dummy_dim), device=device)
    dummy_mask = torch.ones((1,1), dtype=torch.long, device=device)

    for i in tqdm(range(0, len(caps), BATCH_TXT), desc="Text enc"):
        batch_caps = caps[i:i+BATCH_TXT]
        tok = model.tokenizer(batch_caps, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out  = model.text_encoder(input_ids=tok.input_ids,
                                      attention_mask=tok.attention_mask,
                                      encoder_hidden_states=dummy_img,
                                      encoder_attention_mask=dummy_mask,
                                      return_dict=True)
            proj = model.text_proj(out.last_hidden_state[:, 0])
        feats.append(proj.cpu())
    return torch.nn.functional.normalize(torch.cat(feats, 0), dim=-1)

# ---------- main ----------
def main(args):
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    from models.blip_itm import blip_itm
    model = blip_itm(pretrained=args.blip_model_path,
                     image_size=384, vit="large").eval().to(device)

    tfm = transforms.Compose([
        transforms.Resize((384,384), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466,0.4578275,0.40821073),
                             std =(0.26862954,0.26130258,0.27577711))])

    # ---------- load data ----------
    img_names_full = json.load(open(args.blip_image_json))
    caps            = [l.strip() for l in open(args.blip_caption_path, encoding="utf-8")]
    assert len(img_names_full) == len(caps), "mismatch in number of images and captions"
    N, K = len(caps), args.topk

    # <<< 1) （Flickr test split）
    unique_img_names = list(dict.fromkeys(img_names_full))
    name2idx = {n:i for i,n in enumerate(unique_img_names)}
    txt2img  = np.array([name2idx[n] for n in img_names_full], dtype=np.int32)

    # ---------- extract embeddings ----------
    img_feats = encode_images(model, tfm, args.blip_image_dir,
                              unique_img_names, device)
    txt_feats = encode_texts(model, caps, device)

    # ---------- ITC（t2i） ----------
    sim      = (txt_feats @ img_feats.T).numpy()            # 5000 × 1000
    itc_top  = sim.argsort(axis=1)[:, ::-1][:, :K]
    ranks_itc = np.full(N, K, np.int32)
    for i in range(N):
        gt = txt2img[i]
        for rank, idx in enumerate(itc_top[i]):
            if idx == gt:
                ranks_itc[i] = rank
                break

    # ---------- ITM rerank ----------
    ranks_itm = np.full(N, K, np.int32)
    for i in tqdm(range(N), desc="ITM rerank"):
        cand_ids = itc_top[i]
        imgs = torch.cat([
            tfm(Image.open(os.path.join(args.blip_image_dir,
                                         unique_img_names[j])).convert("RGB")
            ).unsqueeze(0) for j in cand_ids]).to(device)

        with torch.no_grad():
            scores = torch.nn.functional.softmax(
                        model(imgs, [caps[i]]*K, match_head="itm"), dim=1)[:,1]
        rerank = cand_ids[scores.argsort(descending=True).cpu()]
        for rank, idx in enumerate(rerank):
            if idx == txt2img[i]:
                ranks_itm[i] = rank
                break

    # ---------- metrics ----------
    res = {
        "Text→Image_ITC":   {f"R@{k}": calc_recall(ranks_itc, k) for k in (1,5,10)},
        "Text→Image_ITC+ITM": {f"R@{k}": calc_recall(ranks_itm, k) for k in (1,5,10)}
    }
    print(json.dumps(res, indent=2, ensure_ascii=False))

# ───────────────────────────────────
if __name__ == "__main__":
    class Args:
        blip_model_path   = "/home/xxx/blip/model_large_retrieval_coco.pth"
        blip_image_json   = "/home/xxx/RQ-VAE/data/coco/emb/test_emb.json"
        blip_image_dir    = "/home/xxx/RQ-VAE/data/coco/images/split/test"
        blip_caption_path = "/home/xxx/data/coco/pseudo_VT_1024-512_1-c1024_e3000_lr0.0001_mse/test.source"
        topk = 10
    main(Args())
