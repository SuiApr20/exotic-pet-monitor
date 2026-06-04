"""
下载 Open Images 数据集（仓鼠 + 鹦鹉）并整理为 YOLO 格式
荷兰猪数据请手动准备

用法:
    python training\download_dataset.py
"""
import subprocess
import sys
import random
import shutil
from pathlib import Path

CLASSES = ["Hamster", "Parrot"]  # 荷兰猪不在 Open Images 中，需单独准备
LIMIT = 300
OIDV4_DIR = Path("../OIDv4_ToolKit").resolve()
DATASET_DIR = Path("dataset")
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


def step1_download():
    print("=" * 60)
    print("📥 步骤1: 从 Open Images 下载数据集")
    print(f"   类别: {', '.join(CLASSES)} (荷兰猪需手动准备)")
    print(f"   每类上限: {LIMIT} 张")
    print("=" * 60)

    cmd = [
        sys.executable,
        str(OIDV4_DIR / "main.py"),
        "downloader",
        "--classes", *CLASSES,
        "--type_csv", "train",
        "--limit", str(LIMIT),
        "--multiclasses", "1",
        "--n_threads", "4",
    ]

    print(f"\n执行: {' '.join(cmd)}\n")
    result = subprocess.run(
        cmd,
        cwd=str(OIDV4_DIR),
        input=b"Y\nY\nY\n",  # 自动确认下载
    )
    if result.returncode != 0:
        print(f"⚠ 部分错误，继续整理已下载的文件...")


def step2_organize():
    print("\n" + "=" * 60)
    print("📦 步骤2: 整理数据集")
    print("=" * 60)

    oid_train = OIDV4_DIR / "OID" / "Dataset" / "train"
    if not oid_train.exists():
        print(f"❌ 找不到 {oid_train}")
        return False

    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (DATASET_DIR / sub).mkdir(parents=True, exist_ok=True)

    classes_file = oid_train / "classes.txt"
    if not classes_file.exists():
        print("❌ 找不到 classes.txt")
        return False

    with open(classes_file, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]
    class_map = {name: i for i, name in enumerate(class_names)}
    print(f"   类别: {class_map}")

    paired = []
    for img_path in sorted(oid_train.glob("*")):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        label_path = oid_train / "Label" / f"{img_path.stem}.txt"
        if label_path.exists():
            paired.append((img_path, label_path))

    random.shuffle(paired)
    split = int(len(paired) * 0.8)
    train_data = paired[:split]
    val_data = paired[split:]

    print(f"   训练集: {len(train_data)}  |  验证集: {len(val_data)}")

    for subset, data in [("train", train_data), ("val", val_data)]:
        img_dir = DATASET_DIR / "images" / subset
        lbl_dir = DATASET_DIR / "labels" / subset
        skipped = 0

        for img_path, lbl_path in data:
            yolo_lines = []
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_name = parts[0]
                    x1, y1, x2, y2 = map(float, parts[1:5])
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1
                    cls_id = class_map.get(cls_name)
                    if cls_id is not None:
                        yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            if not yolo_lines:
                skipped += 1
                continue

            shutil.copy2(img_path, img_dir / img_path.name)
            with open(lbl_dir / lbl_path.name, "w") as f:
                f.write("\n".join(yolo_lines))

        print(f"   {subset}: {len(data) - skipped} 张 (跳过 {skipped})")

    # 生成 data.yaml（预留荷兰猪=2 的位置）
    display = {"Hamster": "仓鼠", "Parrot": "鹦鹉", "Guinea_pig": "荷兰猪"}
    full_path = DATASET_DIR.resolve()

    lines = [
        f"path: {full_path.as_posix()}",
        f"train: images/train",
        f"val: images/val",
        f"nc: 3",
        f"names:",
    ]
    # ID 分配: 0=Hamster, 1=Parrot, 2=Guinea_pig（等数据加入后再用）
    name_map = {"Hamster": 0, "Parrot": 1, "Guinea_pig": 2}
    for name in ["Hamster", "Parrot", "Guinea_pig"]:
        lines.append(f"  {name_map[name]}: {name}  # {display[name]}")

    with open(DATASET_DIR / "data.yaml", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n✅ 仓鼠+鹦鹉数据整理完成!")
    print(f"\n⚠ 荷兰猪(Guinea_pig, class_id=2) 还需手动准备:")
    print(f"   1. 去 Roboflow (universe.roboflow.com) 搜索 'guinea pig'")
    print(f"   2. 下载 YOLO 格式数据集")
    print(f"   3. 把图片放入 dataset/images/train/")
    print(f"   4. 把标签放入 dataset/labels/train/ (class_id=2)")
    print(f"   5. 或用 LabelImg 标注你自己的荷兰猪照片")
    return True


if __name__ == "__main__":
    step1_download()
    step2_organize()
