"""
全局配置文件
"""
import os
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# 模型路径
MODEL_DIR = BASE_DIR / "models"
DETECTION_MODEL = MODEL_DIR / "best.pt"           # YOLO 检测模型（荷兰猪/鹦鹉/仓鼠）
CLASSIFIER_MODEL = MODEL_DIR / "classifier.pth"    # 品种花色分类模型（ResNet）

# 品种花色映射
GUINEA_PIG_COLORS = ["白色", "雕灰", "奶黄", "棕色", "三花"]
GUINEA_PIG_BREEDS = ["长顺", "长逆", "加州", "泰迪"]

# 物种列表
SPECIES = {
    "guinea_pig": "荷兰猪",
    "parrot": "鹦鹉",
    "hamster": "仓鼠"
}

# ROI 默认设置（可在前端覆盖）
DEFAULT_ROI = {
    "water_bottle": {"x": 0, "y": 0, "w": 100, "h": 100},
    "hay_rack": {"x": 0, "y": 0, "w": 100, "h": 100}
}

# 健康判断阈值（参考荷兰猪习性文献）
HEALTH_THRESHOLDS = {
    "eating": {
        "min_visit_duration_sec": 10,     # 在草架区最短停留时间才算进食
        "alert_no_visit_hours": 4,        # 连续N小时未到草架 → 警报
        "alert_low_total_min": 60,        # 24小时内总进食时间少于N分钟 → 警报
    },
    "drinking": {
        "min_visit_duration_sec": 5,      # 在水瓶区最短停留时间
        "alert_no_visit_hours": 6,        # 连续N小时未到水瓶 → 警报
        "alert_low_total_count": 5,       # 24小时内饮水次数少于N次 → 警报
    },
    "sleeping": {
        "max_continuous_idle_min": 120,   # 连续静止超过N分钟 → 异常
        "min_active_ratio_per_hour": 0.1, # 每小时最少活跃时间占比
    },
    "weight": {
        "alert_loss_percent": 10,         # 体重下降超过N% → 警报（需手动录入）
    }
}

# 数据库
DATABASE_URL = f"sqlite:///{BASE_DIR}/backend/database/pet_monitor.db"

# 服务器
HOST = "127.0.0.1"
PORT = 8000

# 上传文件限制（MB）
MAX_UPLOAD_SIZE_MB = 200
