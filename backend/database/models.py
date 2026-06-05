"""
数据库模型定义
使用 SQLAlchemy ORM
"""
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.config import DATABASE_URL

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


class BehaviorEvent(Base):
    """行为事件记录表"""
    __tablename__ = "behavior_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(32), nullable=False, index=True)   # eating / drinking / sleeping
    pig_id = Column(String(32), nullable=False)
    start_time = Column(Float, nullable=False)                     # Unix timestamp
    end_time = Column(Float, nullable=False)
    duration_sec = Column(Float, nullable=False)
    roi_name = Column(String(64), nullable=True)                   # water_bottle / food_bowl
    created_at = Column(DateTime, default=datetime.now)

    def to_dict(self):
        return {
            "id": self.id,
            "event_type": self.event_type,
            "pig_id": self.pig_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_sec": self.duration_sec,
            "roi_name": self.roi_name,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class DailyStats(Base):
    """每日统计汇总表"""
    __tablename__ = "daily_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(16), nullable=False, index=True)          # YYYY-MM-DD
    total_eating_sec = Column(Float, default=0)
    total_drinking_sec = Column(Float, default=0)
    total_sleeping_sec = Column(Float, default=0)
    eating_events = Column(Integer, default=0)
    drinking_events = Column(Integer, default=0)
    health_score = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.now)

    def to_dict(self):
        return {
            "id": self.id,
            "date": self.date,
            "total_eating_sec": self.total_eating_sec,
            "total_drinking_sec": self.total_drinking_sec,
            "total_sleeping_sec": self.total_sleeping_sec,
            "eating_events": self.eating_events,
            "drinking_events": self.drinking_events,
            "health_score": self.health_score,
        }


class DetectionLog(Base):
    """检测记录表（品种识别历史）"""
    __tablename__ = "detection_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_name = Column(String(256), nullable=False)
    species = Column(String(32), nullable=False)                    # 荷兰猪/鹦鹉/仓鼠
    breed = Column(String(64), nullable=True)                       # 长顺/长逆/加州/泰迪
    color = Column(String(32), nullable=True)                       # 白色/雕灰/奶黄/棕色/三花
    confidence = Column(Float, nullable=False)
    bbox = Column(JSON, nullable=True)                              # [x1, y1, x2, y2]
    created_at = Column(DateTime, default=datetime.now)

    def to_dict(self):
        return {
            "id": self.id,
            "image_name": self.image_name,
            "species": self.species,
            "breed": self.breed,
            "color": self.color,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


def init_db():
    """初始化数据库表"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
