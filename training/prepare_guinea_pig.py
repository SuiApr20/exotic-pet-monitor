"""
从三物种数据集中筛选出荷兰猪（class 2）数据，
生成单类数据集用于训练荷兰猪专精模型。

用法:
    python training/prepare_guinea_pig.py
"""
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC_IMAGES = BASE / "dataset/images"
SRC_LABELS = BASE / "dataset/labels"
DST = BASE / "dataset_guinea_pig"
DST_IMAGES = DST / "images"
DST_LABELS = DST / "labels"

# 清空目标目录
if DST.exists():
    shutil.rmtree(DST)

GUINEA_PIG_CLS = 2  # 原数据集中荷兰猪的 class id

for split in ["train", "val"]:
    src_img_dir = SRC_IMAGES / split
    src_lbl_dir = SRC_LABELS / split
    dst_img_dir = DST_IMAGES / split
    dst_lbl_dir = DST_LABELS / split
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    for lbl_file in src_lbl_dir.glob("*.txt"):
        # 读取标签，只保留 class 2 的行
        lines = lbl_file.read_text().strip().splitlines()
        gp_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            if cls_id == GUINEA_PIG_CLS:
                # 重映射 class 2 → 0
                parts[0] = "0"
                gp_lines.append(" ".join(parts))

        if not gp_lines:
            continue  # 这张图没有荷兰猪，跳过

        # 写入新标签
        new_lbl = dst_lbl_dir / lbl_file.name
        new_lbl.write_text("\n".join(gp_lines))

        # 复制图片（找同名图片，支持 jpg/png/jpeg）
        img_name = lbl_file.stem
        img_file = None
        for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG"]:
            candidate = src_img_dir / (img_name + ext)
            if candidate.exists():
                img_file = candidate
                break
        if img_file is None:
            print(f"⚠ 找不到图片: {img_name}")
            continue

        shutil.copy2(img_file, dst_img_dir / img_file.name)
        kept += 1

    print(f"  {split}: 保留 {kept} 张荷兰猪图片")

# 生成 data.yaml
import yaml

yaml_path = DST / "guinea_pig.yaml"
yaml_path.write_text(
    f"path: {DST.as_posix()}\n"
    "train: images/train\n"
    "val: images/val\n"
    "nc: 1\n"
    "names:\n"
    "  0: Guinea_pig\n",
    encoding="utf-8"
)
print(f"\n✅ 荷兰猪专精数据集已生成: {DST}")
print(f"   配置: {yaml_path}")
