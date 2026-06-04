"""
检测相关 API 路由
/upload  - 上传图片进行检测
/detect  - 直接传入 base64 图像检测
/species - 获取物种列表
"""
import base64
import os
import uuid
import cv2
import numpy as np
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.services.yolo_detector import detector
from backend.config import GUINEA_PIG_COLORS, GUINEA_PIG_BREEDS, SPECIES

router = APIRouter(prefix="/api", tags=["detection"])


class Base64Image(BaseModel):
    image_base64: str
    species: str = "guinea_pig"  # 用户选择的物种
    breed: str = ""              # 期望品种（可选）
    color: str = ""              # 期望花色（可选）


@router.get("/species")
async def get_species():
    """获取支持的物种及品种列表"""
    return {
        "species": SPECIES,
        "guinea_pig": {
            "colors": GUINEA_PIG_COLORS,
            "breeds": GUINEA_PIG_BREEDS
        }
    }


@router.post("/upload")
async def upload_image(file: UploadFile = File(...), species: str = Form("guinea_pig")):
    """上传图片文件进行检测"""
    # 验证文件类型
    allowed = {"image/jpeg", "image/png", "image/bmp", "image/webp"}
    if file.content_type not in allowed:
        raise HTTPException(400, f"不支持的文件类型: {file.content_type}")

    # 保存临时文件
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    ext = os.path.splitext(file.filename or "image.jpg")[1] or ".jpg"
    save_path = upload_dir / f"{uuid.uuid4().hex}{ext}"

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    try:
        results = detector.detect_image_file(save_path)
        # 如果选了荷兰猪但没检测到，用分类器兜底
        if species == "guinea_pig" and not any(
            r["class_name"] == "荷兰猪" and r.get("breed_info") for r in results
        ):
            fallback = detector.classify_full_image(
                cv2.imread(str(save_path))
            )
            if fallback:
                results.append(fallback)
    except Exception as e:
        raise HTTPException(500, f"检测失败: {str(e)}")

    # 绘制结果图
    img = cv2.imread(str(save_path))
    annotated = detector.draw_boxes(img, results)
    result_basename = f"result_{save_path.name}"
    result_path = upload_dir / result_basename
    cv2.imwrite(str(result_path), annotated)

    return {
        "success": True,
        "filename": file.filename,
        "detections": results,
        "result_image": str(result_basename),
        "species_selected": SPECIES.get(species, species)
    }


@router.post("/detect-base64")
async def detect_base64(req: Base64Image):
    """通过 Base64 图片进行检测"""
    try:
        # 解码 base64，移除可能的 data:image 前缀
        b64_str = req.image_base64
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        img_bytes = base64.b64decode(b64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("无法解码图片")

        results = detector.detect(img)

        # 绘制结果
        annotated = detector.draw_boxes(img, results)
        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        result_b64 = base64.b64encode(buffer).decode()

        return {
            "success": True,
            "detections": results,
            "result_image_base64": result_b64
        }
    except Exception as e:
        raise HTTPException(500, f"检测失败: {str(e)}")
