"""
FastAPI 应用入口
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.config import HOST, PORT, BASE_DIR, CLASSIFIER_MODEL, AVAILABLE_MODELS
from backend.database.models import init_db
from backend.routers import detection, behavior
from backend.services.yolo_detector import detector

# 创建上传目录
upload_dir = Path(BASE_DIR) / "uploads"
upload_dir.mkdir(exist_ok=True)
(upload_dir / "videos").mkdir(exist_ok=True)
(Path(BASE_DIR) / "models").mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    print(f"🚀 异宠智能监护系统启动中...")
    print(f"   📍 http://{HOST}:{PORT}")
    print(f"   📄 API 文档: http://{HOST}:{PORT}/docs")

    # 预加载所有可用模型
    for model_key in AVAILABLE_MODELS:
        detector._load_yolo(model_key)
    detector.load_classifier(CLASSIFIER_MODEL)
    print(f"   🧠 全部模型已就绪 ({len(AVAILABLE_MODELS)} 个检测模型 + 1 个分类器)")
    yield


app = FastAPI(
    title="异宠智能监护系统",
    description="基于 YOLOv11 的异宠检测识别与健康监护系统",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(detection.router)
app.include_router(behavior.router)

# 静态文件服务
frontend_dir = Path(BASE_DIR) / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
app.mount("/uploads", StaticFiles(directory=str(upload_dir)), name="uploads")


@app.get("/")
async def root():
    """返回前端主页"""
    from fastapi.responses import FileResponse
    return FileResponse(str(Path(BASE_DIR) / "frontend" / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=HOST,
        port=PORT,
        reload=True
    )
