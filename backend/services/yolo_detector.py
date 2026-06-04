"""
YOLO 目标检测 + 品种分类服务
"""
import torch
from ultralytics import YOLO
from pathlib import Path
from typing import Optional
import numpy as np
import cv2
from torchvision import transforms, models
import torch.nn as nn

from backend.config import DETECTION_MODEL, CLASSIFIER_MODEL, SPECIES

# 类别映射：与训练时的 data.yaml 一致
CLASS_NAMES = {
    0: "仓鼠",
    1: "鹦鹉",
    2: "荷兰猪"
}


class BreedClassifier:
    """荷兰猪品种花色分类器（MobileNetV3）"""

    def __init__(self, model_path: Path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.classes = []
        self.img_size = 224
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        if model_path.exists():
            self._load(model_path)

    def _load(self, model_path: Path):
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        self.classes = ckpt['classes']
        self.img_size = ckpt.get('img_size', 224)

        self.model = models.mobilenet_v3_small(weights=None)
        num_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, len(self.classes))
        )
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"品种分类器已加载: {len(self.classes)} 类 {self.classes}")

    def classify_full(self, image: np.ndarray) -> Optional[dict]:
        """直接对全图分类（不裁剪）"""
        if self.model is None:
            return None
        tensor = self.transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

            if conf[0] < 0.6:
                return None

            pred_name = self.classes[pred[0]]
            color, breed = self._parse_name(pred_name)
            return {
                "breed": breed, "color": color,
                "full_name": pred_name,
                "confidence": round(conf[0].item(), 4)
            }

    def classify(self, image: np.ndarray, bbox: list) -> dict:
        """
        裁剪检测框区域并分类品种+花色
        返回: {breed, color, confidence}
        """
        if self.model is None:
            return {"breed": "未知", "color": "未知", "full_name": "未知", "confidence": 0}

        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return {"breed": "未知", "color": "未知", "full_name": "未知", "confidence": 0}

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return {"breed": "未知", "color": "未知", "full_name": "未知", "confidence": 0}

        tensor = self.transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            pred_name = self.classes[pred[0]]
            confidence = conf[0].item()

        color, breed = self._parse_name(pred_name)
        return {
            "breed": breed, "color": color,
            "full_name": pred_name,
            "confidence": round(confidence, 4)
        }

    def _parse_name(self, pred_name: str) -> tuple:
        """解析 '猪1_雕灰加州' -> ('雕灰', '加州')"""
        name_parts = pred_name.split("_", 1)
        if len(name_parts) >= 2:
            attr = name_parts[1]
            colors = ["白色", "雕灰", "奶黄", "棕色", "三花"]
            breeds = ["长顺", "长逆", "加州", "泰迪"]
            color = "未知"; breed = "未知"
            for c in colors:
                if attr.startswith(c):
                    color = c
                    breed = attr[len(c):]
                    break
            for b in breeds:
                if attr.endswith(b):
                    breed = b
                    color = attr[:-len(b)]
                    break
            return color, breed
        return "未知", "未知"


class YOLODetector:
    """YOLO 检测器单例"""

    _instance: Optional["YOLODetector"] = None
    _model: Optional[YOLO] = None
    _classifier: Optional[BreedClassifier] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self, model_path: Optional[Path] = None) -> YOLO:
        path = model_path or DETECTION_MODEL
        if self._model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if path.exists():
                self._model = YOLO(str(path))
                print(f"YOLO 模型已加载: {path} (device={device})")
            else:
                print(f"⚠ 模型文件 {path} 不存在，使用 YOLOv11n 预训练权重")
                self._model = YOLO("yolo11n.pt")
        return self._model

    def load_classifier(self, path: Optional[Path] = None):
        path = path or CLASSIFIER_MODEL
        self._classifier = BreedClassifier(path)

    def classify_full_image(self, image: np.ndarray) -> Optional[dict]:
        """直接对全图做品种分类（无需 YOLO 检测）"""
        if self._classifier is None or self._classifier.model is None:
            return None
        result = self._classifier.classify_full(image)
        if result and result.get("confidence", 0) > 0.5:
            h, w = image.shape[:2]
            cx, cy = w / 2, h / 2
            bw, bh = w * 0.6, h * 0.6
            breed_conf = result.get("confidence", 0)
            return {
                "class_name": "荷兰猪",
                "class_id": 2,
                "confidence": breed_conf,
                "bbox": [round(cx - bw/2), round(cy - bh/2), round(cx + bw/2), round(cy + bh/2)],
                "breed_info": {
                    "breed": result.get("breed", "未知"),
                    "color": result.get("color", "未知"),
                    "full_name": result.get("full_name", "未知"),
                    "confidence": breed_conf,
                }
            }
        return None

    def detect(self, image: np.ndarray, conf_threshold: float = 0.25,
               fallback_classify: bool = False) -> list[dict]:
        model = self.load_model()
        results = model(image, conf=conf_threshold, verbose=False)

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()

                    det = {
                        "class_id": cls_id,
                        "class_name": CLASS_NAMES.get(cls_id, f"未知({cls_id})"),
                        "confidence": round(conf, 4),
                        "bbox": [round(v, 1) for v in xyxy],
                        "breed_info": None
                    }

                    # 如果是荷兰猪，用分类器识别品种花色
                    if cls_id == 2 and self._classifier is not None:
                        det["breed_info"] = self._classifier.classify(image, xyxy)

                    detections.append(det)

        return detections

    def detect_image_file(self, image_path: Path, conf_threshold: float = 0.25) -> list[dict]:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        return self.detect(img, conf_threshold)

    @staticmethod
    def draw_boxes(image: np.ndarray, detections: list[dict]) -> np.ndarray:
        img = image.copy()
        colors = {"荷兰猪": (0, 255, 0), "鹦鹉": (255, 0, 0), "仓鼠": (0, 0, 255)}

        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            cls_name = det["class_name"]
            conf = det["confidence"]
            color = colors.get(cls_name, (255, 255, 255))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 构建标签
            if det.get("breed_info") and det["breed_info"].get("full_name") != "未知":
                bi = det["breed_info"]
                label = f"{bi.get('color','')}{bi.get('breed','')} ({conf:.0%})"
            else:
                label = f"{cls_name} {conf:.2f}"

            cv2.putText(img, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return img


detector = YOLODetector()

