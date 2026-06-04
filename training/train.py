"""
YOLOv11 训练脚本
用于异宠物种检测（荷兰猪/鹦鹉/仓鼠）

用法:
    cd exotic-pet-monitor
    ../venv/Scripts/activate
    python training/train.py

    或指定参数:
    python training/train.py --model yolo11s.pt --epochs 100 --batch 16
"""
import argparse
import sys
from pathlib import Path

# 将项目根目录加入 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLOv11 异宠检测训练")
    parser.add_argument("--model", default="yolo11n.pt", help="预训练模型 (yolo11n.pt / yolo11s.pt / yolo11m.pt)")
    parser.add_argument("--data", default="dataset/data.yaml", help="数据集配置文件路径")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="批次大小 (RTX 3060 6GB 建议 8-16)")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")
    parser.add_argument("--lr", type=float, default=0.001, help="初始学习率")
    parser.add_argument("--patience", type=int, default=15, help="早停耐心值")
    parser.add_argument("--device", default="0", help="设备: 0=GPU, cpu=CPU")
    parser.add_argument("--resume", action="store_true", help="从上次中断恢复训练")
    parser.add_argument("--freeze", type=int, default=10, help="冻结骨干网络层数 (迁移学习)")
    args = parser.parse_args()

    data_yaml = Path(__file__).parent / args.data
    if not data_yaml.exists():
        print(f"❌ 数据集配置文件不存在: {data_yaml}")
        print("   请先准备数据集并修改 data.yaml 中的路径")
        sys.exit(1)

    print("=" * 60)
    print(f"🚀 开始训练 YOLOv11")
    print(f"   模型: {args.model}")
    print(f"   数据: {args.data}")
    print(f"   轮数: {args.epochs} | Batch: {args.batch} | 尺寸: {args.imgsz}")
    print(f"   学习率: {args.lr} | 早停: {args.patience} | 冻结层: {args.freeze}")
    print("=" * 60)

    # 加载模型
    model = YOLO(args.model)

    # 训练参数
    train_args = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr,
        patience=args.patience,
        device=args.device,
        resume=args.resume,
        freeze=args.freeze,
        # 数据增强
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # 验证
        val=True,
        # 保存
        save=True,
        save_period=10,
        project="runs/train",
        name="exotic_pet_detection",
        exist_ok=True,
        # 日志
        verbose=True,
    )

    # 开始训练
    results = model.train(**train_args)

    print("=" * 60)
    print("✅ 训练完成!")
    print(f"   最佳模型保存在: runs/train/exotic_pet_detection/weights/best.pt")

    # 评估
    print("\n📊 模型评估:")
    metrics = model.val()
    print(f"   mAP50: {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")

    # 导出最佳模型到 models/ 目录
    best_pt = Path("runs/train/exotic_pet_detection/weights/best.pt")
    if best_pt.exists():
        import shutil
        target = Path("../models/best.pt")
        shutil.copy(best_pt, target)
        print(f"   模型已复制到: models/best.pt")

    print("=" * 60)


if __name__ == "__main__":
    main()
