"""
ROI 碰撞检测引擎
支持两种模式：
  1. 行为模型：直接检测 食盆/水壶 与荷兰猪的位置关系
  2. 通用模型：基于 ROI 区域的碰撞检测
"""
import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from backend.config import HEALTH_THRESHOLDS

# 行为模型可识别的荷兰猪类别名
PIG_CLASS_NAMES = {"奶黄加州", "三花美泰", "白加州", "荷兰猪"}

# 行为相关物品
EATING_OBJECTS = {"食盆"}       # 检测到猪靠近食盆 → 进食
DRINKING_OBJECTS = {"水壶"}     # 检测到猪靠近水壶 → 饮水


@dataclass
class ROIZone:
    """ROI 区域定义"""
    name: str                    # 名称：water_bottle / food_bowl
    x: float
    y: float
    width: float
    height: float

    @property
    def box(self) -> tuple[float, float, float, float]:
        """返回 (x1, y1, x2, y2)"""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def to_dict(self) -> dict:
        return {"name": self.name, "x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass
class PigTracker:
    """单个荷兰猪的行为追踪"""
    pig_id: str
    current_position: Optional[tuple[float, float, float, float]] = None
    last_moved_time: float = field(default_factory=time.time)
    idle_start_time: Optional[float] = None
    water_start_time: Optional[float] = None
    hay_start_time: Optional[float] = None
    water_miss_count: int = 0           # 连续未检测到饮水的帧数
    hay_miss_count: int = 0             # 连续未检测到进食的帧数


class ROIEngine:
    """ROI 碰撞检测引擎"""

    def __init__(self):
        self.roi_zones: list[ROIZone] = []
        self.trackers: dict[str, PigTracker] = {}
        self.event_log: list[dict] = []  # 行为事件记录
        self.stats: dict = {
            "total_eating_sec": 0.0,
            "total_drinking_sec": 0.0,
            "total_sleeping_sec": 0.0,
            "eating_events": 0,
            "drinking_events": 0,
            "last_eating_time": None,
            "last_drinking_time": None,
        }

    def set_roi_zones(self, zones: list[dict]):
        """从前端设置的 ROI 更新"""
        self.roi_zones = [ROIZone(**z) for z in zones]

    @staticmethod
    def contains_center(big_box: tuple, small_box: tuple) -> bool:
        """判断 small_box 的中心点是否在 big_box 内（适用于大小悬殊的物体）"""
        cx = (small_box[0] + small_box[2]) / 2
        cy = (small_box[1] + small_box[3]) / 2
        return big_box[0] <= cx <= big_box[2] and big_box[1] <= cy <= big_box[3]

    def compute_iou(self, box_a: tuple, box_b: tuple) -> float:
        """计算两个边界框的 IoU (Intersection over Union)"""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union_area = area_a + area_b - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def update(self, detections: list[dict], frame_timestamp: Optional[float] = None) -> dict:
        """
        每帧调用一次，更新行为追踪状态。
        自动检测是行为模型还是通用模型，采用不同判定策略。

        行为模型：检测 食盆/水壶 与猪的位置关系
        通用模型：检测猪是否在 ROI 区域内
        """
        now = frame_timestamp or time.time()
        events = []
        alerts = []

        # 分离猪、进食物品、饮水物品
        pigs = [d for d in detections if d["class_name"] in PIG_CLASS_NAMES]
        salt_objects = [d for d in detections if d["class_name"] in EATING_OBJECTS]
        water_objects = [d for d in detections if d["class_name"] in DRINKING_OBJECTS]

        # 检测模式判断：有食盆/水壶检测结果 → 行为模型模式
        has_behavior_objects = bool(salt_objects or water_objects)

        tracked_ids = set()

        for i, pig in enumerate(pigs):
            pig_id = f"pig_{i}"
            tracked_ids.add(pig_id)
            bbox = tuple(pig["bbox"])

            if pig_id not in self.trackers:
                self.trackers[pig_id] = PigTracker(pig_id=pig_id)

            tracker = self.trackers[pig_id]
            # 先保存旧位置，再更新（顺序很重要）
            prev_pos = tracker.current_position
            tracker.current_position = bbox

            # 判断是否在移动
            is_moving = False
            if prev_pos is not None:
                prev_cx = (prev_pos[0] + prev_pos[2]) / 2
                prev_cy = (prev_pos[1] + prev_pos[3]) / 2
                cur_cx = (bbox[0] + bbox[2]) / 2
                cur_cy = (bbox[1] + bbox[3]) / 2
                displacement = np.sqrt((cur_cx - prev_cx) ** 2 + (cur_cy - prev_cy) ** 2)
                is_moving = displacement > 15

            if is_moving:
                tracker.last_moved_time = now
                tracker.idle_start_time = None

            in_water = False
            in_hay = False

            if has_behavior_objects:
                # === 行为模型模式：物体中心点是否在猪的检测框内 ===
                # （比 IoU 更适合大小悬殊的物体，比如猪 200×200 vs 食盆 40×40）
                for w_obj in water_objects:
                    if self.contains_center(bbox, tuple(w_obj["bbox"])):
                        in_water = True
                        break
                for s_obj in salt_objects:
                    if self.contains_center(bbox, tuple(s_obj["bbox"])):
                        in_hay = True
                        break
            else:
                # === 通用模型模式：ROI 区域碰撞 ===
                for roi in self.roi_zones:
                    iou = self.compute_iou(bbox, roi.box)
                    if iou > 0.2:
                        if roi.name == "water_bottle":
                            in_water = True
                        elif roi.name == "food_bowl":
                            in_hay = True

            # === 饮水判定（容忍 3 帧闪烁不重置计时器）===
            if in_water:
                tracker.water_miss_count = 0
                if tracker.water_start_time is None:
                    tracker.water_start_time = now
            else:
                tracker.water_miss_count += 1
                if tracker.water_miss_count > 3 and tracker.water_start_time is not None:
                    duration = now - tracker.water_start_time
                    if duration >= 5:
                        events.append({
                            "type": "drinking", "pig_id": pig_id,
                            "duration_sec": round(duration, 1)
                        })
                        self.stats["total_drinking_sec"] += duration
                        self.stats["drinking_events"] += 1
                        self.stats["last_drinking_time"] = now
                    tracker.water_start_time = None
                    tracker.water_miss_count = 0

            # === 进食判定（容忍 3 帧闪烁）===
            if in_hay:
                tracker.hay_miss_count = 0
                if tracker.hay_start_time is None:
                    tracker.hay_start_time = now
            else:
                tracker.hay_miss_count += 1
                if tracker.hay_miss_count > 3 and tracker.hay_start_time is not None:
                    duration = now - tracker.hay_start_time
                    if duration >= 10:
                        events.append({
                            "type": "eating", "pig_id": pig_id,
                            "duration_sec": round(duration, 1)
                        })
                        self.stats["total_eating_sec"] += duration
                        self.stats["eating_events"] += 1
                        self.stats["last_eating_time"] = now
                    tracker.hay_start_time = None
                    tracker.hay_miss_count = 0

            # === 静止判定 ===
            if not is_moving and not in_water and not in_hay:
                if tracker.idle_start_time is None:
                    tracker.idle_start_time = now
                else:
                    idle_duration = now - tracker.idle_start_time
                    idle_limit = HEALTH_THRESHOLDS["sleeping"].get("max_continuous_idle_sec", 300)
                    if idle_duration > idle_limit:
                        alerts.append({
                            "severity": "high",
                            "type": "prolonged_idle",
                            "pig_id": pig_id,
                            "duration_min": round(idle_duration / 60, 1),
                            "message": f"{pig_id} 已静止 {idle_duration/60:.0f} 分钟",
                            "suggestion": "请检查异宠状态，确认是否健康"
                        })

        self.trackers = {k: v for k, v in self.trackers.items() if k in tracked_ids}

        # 健康警报
        if self.stats["last_eating_time"] is not None:
            hours_since_eating = (now - self.stats["last_eating_time"]) / 3600
            if hours_since_eating > 4:
                alerts.append({
                    "type": "no_eating",
                    "hours": round(hours_since_eating, 1),
                    "message": f"已 {hours_since_eating:.1f} 小时未进食，请检查！"
                })
        if self.stats["last_drinking_time"] is not None:
            hours_since_drinking = (now - self.stats["last_drinking_time"]) / 3600
            if hours_since_drinking > 6:
                alerts.append({
                    "type": "no_drinking",
                    "hours": round(hours_since_drinking, 1),
                    "message": f"已 {hours_since_drinking:.1f} 小时未饮水，请检查！"
                })

        self.event_log.extend(events)

        return {
            "status": "ok",
            "events_triggered": events,
            "health_alerts": alerts,
            "stats_snapshot": {
                "total_eating_sec": round(self.stats["total_eating_sec"], 1),
                "total_drinking_sec": round(self.stats["total_drinking_sec"], 1),
                "eating_events": self.stats["eating_events"],
                "drinking_events": self.stats["drinking_events"],
            }
        }

    def get_summary(self) -> dict:
        """获取当前统计摘要"""
        return {
            "stats": self.stats,
            "roi_zones": [z.to_dict() for z in self.roi_zones],
            "active_pigs": len(self.trackers),
            "total_events": len(self.event_log)
        }

    def reset(self):
        """重置所有统计"""
        self.trackers.clear()
        self.event_log.clear()
        self.stats = {
            "total_eating_sec": 0.0,
            "total_drinking_sec": 0.0,
            "total_sleeping_sec": 0.0,
            "eating_events": 0,
            "drinking_events": 0,
            "last_eating_time": None,
            "last_drinking_time": None,
        }


# 全局单例
roi_engine = ROIEngine()
