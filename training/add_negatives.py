"""给鹦鹉数据集加负样本（荷兰猪图片，空标签=无鹦鹉）"""
import shutil
from pathlib import Path

PARROT = Path("C:/Users/Sui/Desktop/xxxs/exotic-pet-monitor/dataset/parrot")
GUINEA = Path("C:/Users/Sui/Desktop/xxxs/exotic-pet-monitor/training/dataset_guinea_pig")

for split, count in [("train", 50), ("val", 20)]:
    src_imgs = list((GUINEA / "images" / split).glob("*"))[:count]
    dst_img = PARROT / "images" / split
    dst_lbl = PARROT / "labels" / split

    added = 0
    for img in src_imgs:
        if not img.is_file():
            continue
        target = dst_img / f"neg_gp_{img.name}"
        if not target.exists():
            shutil.copy2(img, target)
            (dst_lbl / f"neg_gp_{img.stem}.txt").write_text("")
            added += 1

    print(f"  {split}: 添加 {added} 张负样本")

print("\n完成。")
