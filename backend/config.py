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

# 可选模型注册表（前端可切换）
AVAILABLE_MODELS = {
    "best": {
        "key": "best",
        "name": "通用检测",
        "path": MODEL_DIR / "best.pt",
        "description": "三物种通用检测（荷兰猪/鹦鹉/仓鼠）",
        "class_names": {0: "仓鼠", 1: "鹦鹉", 2: "荷兰猪"},
        "has_breed_classifier": True,
    },
    "guinea_pig": {
        "key": "guinea_pig",
        "name": "荷兰猪专精",
        "path": MODEL_DIR / "guinea_pig.pt",
        "description": "荷兰猪专用检测，精度更高",
        "class_names": {0: "荷兰猪"},
        "has_breed_classifier": True,
    },
    "parrot": {
        "key": "parrot",
        "name": "鹦鹉专精",
        "path": MODEL_DIR / "parrot.pt",
        "description": "鹦鹉专用检测",
        "class_names": {0: "鹦鹉"},
        "has_breed_classifier": False,
    },
    "behavior": {
        "key": "behavior",
        "name": "行为监测",
        "path": MODEL_DIR / "behavior.pt",
        "description": "进食饮水行为检测（食盆/水壶/品种）",
        "class_names": {0: "奶黄加州", 1: "三花美泰", 2: "白加州", 3: "食盆", 4: "水壶"},
        "has_breed_classifier": False,
        "for_monitor": True,
    },
}

# 默认模型
DEFAULT_MODEL_KEY = "best"

# 品种花色映射
GUINEA_PIG_COLORS = ["奶黄", "白色", "三花"]
GUINEA_PIG_BREEDS = ["加州", "美泰"]

# 物种列表
SPECIES = {
    "guinea_pig": "荷兰猪",
    "parrot": "鹦鹉",
    "hamster": "仓鼠"
}

# ROI 默认设置（可在前端覆盖）
DEFAULT_ROI = {
    "water_bottle": {"x": 0, "y": 0, "w": 100, "h": 100},
    "food_bowl": {"x": 0, "y": 0, "w": 100, "h": 100}
}

# 健康判断阈值（参考荷兰猪习性文献）
HEALTH_THRESHOLDS = {
    "eating": {
        "min_visit_duration_sec": 10,     # 在食盆区最短停留时间才算进食
        "alert_no_visit_hours": 4,        # 连续N小时未到食盆 → 警报
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
        "max_continuous_idle_sec": 300,   # 连续静止超过N秒触发警报（测试用，正式改为1800即30分钟）
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
