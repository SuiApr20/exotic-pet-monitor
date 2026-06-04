"""
ROI 碰撞检测引擎
判断荷兰猪是否在指定区域内（水瓶、草架），统计停留时间
"""
import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class ROIZone:
    """ROI 区域定义"""
    name: str                    # 名称：water_bottle / hay_rack
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
    current_position: Optional[tuple[float, float, float, float]] = None  # 当前 bbox
    last_moved_time: float = field(default_factory=time.time)              # 上次移动时间
    idle_start_time: Optional[float] = None                                 # 开始静止的时间
    water_start_time: Optional[float] = None                                # 开始在水瓶区的时间
    hay_start_time: Optional[float] = None                                  # 开始在草架区的时间


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
        每帧调用一次，更新行为追踪状态

        参数:
            detections: YOLO 检测结果列表
            frame_timestamp: 帧时间戳（秒）

        返回:
            {status, events_triggered, health_alerts}
        """
        now = frame_timestamp or time.time()
        events = []
        alerts = []

        # 只追踪荷兰猪
        pigs = [d for d in detections if d["class_name"] == "荷兰猪"]

        tracked_ids = set()

        for i, pig in enumerate(pigs):
            pig_id = f"pig_{i}"
            tracked_ids.add(pig_id)
            bbox = tuple(pig["bbox"])  # (x1, y1, x2, y2)

            if pig_id not in self.trackers:
                self.trackers[pig_id] = PigTracker(pig_id=pig_id)

            tracker = self.trackers[pig_id]
            prev_pos = tracker.current_position
            tracker.current_position = bbox

            # 判断是否在移动（bbox 中心点位移 > 阈值）
            is_moving = False
            if prev_pos is not None:
                prev_cx = (prev_pos[0] + prev_pos[2]) / 2
                prev_cy = (prev_pos[1] + prev_pos[3]) / 2
                cur_cx = (bbox[0] + bbox[2]) / 2
                cur_cy = (bbox[1] + bbox[3]) / 2
                displacement = np.sqrt((cur_cx - prev_cx) ** 2 + (cur_cy - prev_cy) ** 2)
                is_moving = displacement > 15  # 像素阈值

            if is_moving:
                tracker.last_moved_time = now
                tracker.idle_start_time = None

            # 检查与 ROI 区域的重叠
            in_water = False
            in_hay = False

            for roi in self.roi_zones:
                iou = self.compute_iou(bbox, roi.box)
                if iou > 0.2:  # 重叠阈值
                    if roi.name == "water_bottle":
                        in_water = True
                    elif roi.name == "hay_rack":
                        in_hay = True

            # === 饮水判定 ===
            if in_water and tracker.water_start_time is None:
                tracker.water_start_time = now
            elif not in_water and tracker.water_start_time is not None:
                duration = now - tracker.water_start_time
                if duration >= 5:  # 至少停留5秒才算一次饮水
                    events.append({
                        "type": "drinking",
                        "pig_id": pig_id,
                        "start_time": tracker.water_start_time,
                        "end_time": now,
                        "duration_sec": round(duration, 1)
                    })
                    self.stats["total_drinking_sec"] += duration
                    self.stats["drinking_events"] += 1
                    self.stats["last_drinking_time"] = now
                tracker.water_start_time = None

            # === 进食判定 ===
            if in_hay and tracker.hay_start_time is None:
                tracker.hay_start_time = now
            elif not in_hay and tracker.hay_start_time is not None:
                duration = now - tracker.hay_start_time
                if duration >= 10:  # 至少停留10秒才算一次进食
                    events.append({
                        "type": "eating",
                        "pig_id": pig_id,
                        "start_time": tracker.hay_start_time,
                        "end_time": now,
                        "duration_sec": round(duration, 1)
                    })
                    self.stats["total_eating_sec"] += duration
                    self.stats["eating_events"] += 1
                    self.stats["last_eating_time"] = now
                tracker.hay_start_time = None

            # === 静止/睡觉判定 ===
            if not is_moving and not in_water and not in_hay:
                if tracker.idle_start_time is None:
                    tracker.idle_start_time = now
                else:
                    idle_duration = now - tracker.idle_start_time
                    if idle_duration > 30 * 60:  # 静止超过30分钟 → 可能睡觉
                        alerts.append({
                            "type": "prolonged_idle",
                            "pig_id": pig_id,
                            "duration_min": round(idle_duration / 60, 1),
                            "message": f"{pig_id} 已静止 {idle_duration/60:.0f} 分钟，可能异常"
                        })

        # 清理不再存在的猪的追踪器
        self.trackers = {k: v for k, v in self.trackers.items() if k in tracked_ids}

        # === 健康警报检查 ===
        if self.stats["last_eating_time"] is not None:
            hours_since_eating = (now - self.stats["last_eating_time"]) / 3600
            if hours_since_eating > 4:
                alerts.append({
                    "type": "no_eating",
                    "hours": round(hours_since_eating, 1),
                    "message": f"荷兰猪已 {hours_since_eating:.1f} 小时未进食，请检查！"
                })

        if self.stats["last_drinking_time"] is not None:
            hours_since_drinking = (now - self.stats["last_drinking_time"]) / 3600
            if hours_since_drinking > 6:
                alerts.append({
                    "type": "no_drinking",
                    "hours": round(hours_since_drinking, 1),
                    "message": f"荷兰猪已 {hours_since_drinking:.1f} 小时未饮水，请检查！"
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
