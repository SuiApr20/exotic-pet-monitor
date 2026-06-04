"""
行为监护 API 路由
/roi/set       - 设置 ROI 区域
/behavior/stats - 获取行为统计
/behavior/health - 获取健康评估
/video/analyze  - 上传视频进行分析
/ws/monitor     - WebSocket 实时监控
"""
import time
import cv2
import numpy as np
import asyncio
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel

from backend.services.yolo_detector import detector
from backend.services.roi_engine import roi_engine
from backend.services.health_analyzer import health_analyzer

router = APIRouter(prefix="/api", tags=["behavior"])


class ROISetRequest(BaseModel):
    zones: list[dict]  # [{name, x, y, width, height}, ...]


@router.post("/roi/set")
async def set_roi(req: ROISetRequest):
    """设置 ROI 区域"""
    roi_engine.set_roi_zones(req.zones)
    return {
        "success": True,
        "roi_zones": [z.to_dict() for z in roi_engine.roi_zones]
    }


@router.get("/roi/get")
async def get_roi():
    """获取当前 ROI 设置"""
    return {
        "roi_zones": [z.to_dict() for z in roi_engine.roi_zones]
    }


@router.get("/behavior/stats")
async def get_stats():
    """获取行为统计"""
    return roi_engine.get_summary()


@router.get("/behavior/health")
async def get_health(hours: float = 24):
    """获取健康评估"""
    summary = roi_engine.get_summary()
    stats = summary["stats"]
    result = health_analyzer.assess(stats, hours)
    return result


@router.post("/behavior/reset")
async def reset_stats():
    """重置统计数据"""
    roi_engine.reset()
    return {"success": True, "message": "统计数据已重置"}


@router.post("/video/analyze")
async def analyze_video(file: UploadFile = File(...), sample_rate: int = 2):
    """
    上传视频文件进行分析

    参数:
        file: 视频文件
        sample_rate: 每秒采样帧数（默认2帧/秒，减少计算量）
    """
    # 保存视频
    video_dir = Path("uploads/videos")
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"temp_{file.filename}"
    content = await file.read()
    with open(video_path, "wb") as f:
        f.write(content)

    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    if duration_sec == 0:
        raise HTTPException(400, "无法读取视频时长")

    # 重置统计
    roi_engine.reset()

    frame_interval = max(1, int(fps / sample_rate))
    frame_count = 0
    processed = 0

    timeline = []  # 时间线数据，用于前端图表

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            detections = detector.detect(frame, conf_threshold=0.3)
            result = roi_engine.update(detections, frame_timestamp=timestamp)
            processed += 1

            if result["events_triggered"]:
                for evt in result["events_triggered"]:
                    timeline.append({
                        "time_sec": round(timestamp, 1),
                        "type": evt["type"],
                        "duration_sec": evt["duration_sec"]
                    })

        frame_count += 1

    cap.release()

    # 清理
    try:
        video_path.unlink()
    except Exception:
        pass

    summary = roi_engine.get_summary()
    health = health_analyzer.assess(summary["stats"], duration_sec / 3600)

    return {
        "success": True,
        "video_duration_sec": round(duration_sec, 1),
        "frames_processed": processed,
        "fps": round(fps, 1),
        "stats": summary["stats"],
        "health": health,
        "timeline": timeline
    }


@router.websocket("/ws/monitor")
async def websocket_monitor(websocket: WebSocket):
    """
    WebSocket 实时监控
    客户端发送 base64 编码的帧，服务端返回检测结果
    """
    import base64

    await websocket.accept()
    print("WebSocket 监控连接已建立")

    roi_engine.reset()
    last_frame_time = time.time()

    try:
        while True:
            data = await websocket.receive_json()
            frame_b64 = data.get("frame", "")
            roi_zones = data.get("roi_zones", None)

            if roi_zones:
                roi_engine.set_roi_zones(roi_zones)

            # 解码帧
            if "," in frame_b64:
                frame_b64 = frame_b64.split(",")[1]
            img_bytes = base64.b64decode(frame_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_json({"error": "无法解码帧"})
                continue

            # 检测
            now = time.time()
            detections = detector.detect(frame, conf_threshold=0.3)
            behavior = roi_engine.update(detections, frame_timestamp=now)

            # 绘制检测框和 ROI
            annotated = detector.draw_boxes(frame, detections)

            # 绘制 ROI 区域
            for zone in roi_engine.roi_zones:
                color = (255, 0, 0) if zone.name == "water_bottle" else (0, 255, 0)
                x, y, w, h = int(zone.x), int(zone.y), int(zone.width), int(zone.height)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                cv2.putText(annotated, zone.name,
                            (x, max(y - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 编码结果
            _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            result_b64 = base64.b64encode(buffer).decode()

            await websocket.send_json({
                "success": True,
                "result_image_base64": result_b64,
                "detections": detections,
                "behavior": behavior,
                "fps_processing": round(1.0 / max(now - last_frame_time, 0.001), 1)
            })

            last_frame_time = now

    except WebSocketDisconnect:
        print("WebSocket 监控连接断开")
    except Exception as e:
        print(f"WebSocket 错误: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
