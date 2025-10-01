import os
import json
import shutil
from tqdm import tqdm
from collections import defaultdict

def split_images_by_caption_json(all_images_dir, caption_json_path, output_root):
    # 1. 载入 JSON
    with open(caption_json_path, 'r') as f:
        data = json.load(f)

    # 2. 解析出 {img_name: split} 映射
    img2split = {}

    # 格式 A：顶层就是图片名
    if all(isinstance(v, dict) and 'caption' in v for v in list(data.values())[:20]):
        for img_name, info in data.items():
            split = info.get('split')
            if split is not None:
                img2split[img_name] = split
    # 格式 B：顶层就是 split
    elif set(data.keys()) >= {'train', 'val', 'test'}:
        for split, split_dict in data.items():
            for img_name in split_dict.keys():
                img2split[img_name] = split
    else:
        raise ValueError("无法识别 JSON 结构，请手动检查。")

    # 3. 创建输出目录
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_root, split), exist_ok=True)

    # 4. 复制 / 移动文件
    missing, copied = defaultdict(list), 0
    for img_name, split in tqdm(img2split.items(), desc="Copying"):
        src = os.path.join(all_images_dir, img_name)
        dst = os.path.join(output_root, split, img_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)   # 如需移动改为 shutil.move
            copied += 1
        else:
            missing[split].append(img_name)

    # 5. 报告结果
    print(f"\n✅ 已处理 {copied} 张图片。")
    if missing:
        total_missing = sum(len(v) for v in missing.values())
        print(f"⚠️ 有 {total_missing} 张图片在源目录中找不到，按 split 统计：")
        for s, lst in missing.items():
            print(f"  {s}: {len(lst)}")
        print("  示例缺失文件:", next(iter(missing.values()))[:5])

if __name__ == '__main__':
    all_images_dir     = 'data/flickr/images/flickr30k-images'          # 所有图片所在目录
    caption_json_path  = 'data/flickr/flickr_split_captions.json'
    output_root        = 'data/flickr/images/split'        # 输出根目录
    split_images_by_caption_json(all_images_dir, caption_json_path, output_root)
