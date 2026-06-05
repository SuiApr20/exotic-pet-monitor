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

from backend.config import DETECTION_MODEL, CLASSIFIER_MODEL, SPECIES, AVAILABLE_MODELS, DEFAULT_MODEL_KEY, GUINEA_PIG_COLORS, GUINEA_PIG_BREEDS


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
        """
        解析分类器输出的类别名 → (花色, 品种)
        支持格式:
          - "奶黄加州" → ("奶黄", "加州")
          - "白加州"   → ("白色", "加州")
          - "三花美泰" → ("三花", "美泰")
          - 旧格式 "猪1_奶黄加州" 也能处理
        """
        # 去掉可能的前缀（如 "猪1_"）
        name = pred_name.split("_", 1)[-1] if "_" in pred_name else pred_name

        # 按长度降序排列颜色名，避免短名误匹配（"白色" 先于 "白"）
        colors_sorted = sorted(GUINEA_PIG_COLORS, key=len, reverse=True)
        breeds_sorted = sorted(GUINEA_PIG_BREEDS, key=len, reverse=True)

        # 尝试匹配 颜色+品种 组合
        for color in colors_sorted:
            if name.startswith(color):
                breed_candidate = name[len(color):]
                for breed in breeds_sorted:
                    if breed_candidate == breed:
                        return color, breed
                # 没匹配到品种，只返回颜色
                return color, breed_candidate if breed_candidate else "未知"

        # 尝试从尾部匹配品种
        for breed in breeds_sorted:
            if name.endswith(breed):
                color_candidate = name[:-len(breed)]
                # 标准化短颜色名 → 全名（"白" → "白色"）
                for c in colors_sorted:
                    if c.startswith(color_candidate) or color_candidate.startswith(c):
                        return c, breed
                return color_candidate if color_candidate else "未知", breed

        # 无法拆分，整名当花色返回
        return name, "未知"


class YOLODetector:
    """YOLO 检测器单例 — 支持多模型切换"""

    _instance: Optional["YOLODetector"] = None
    _models: dict = {}                      # 多模型缓存: {key: YOLO}
    _classifier: Optional[BreedClassifier] = None
    _current_model_key: str = DEFAULT_MODEL_KEY

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def current_model_key(self) -> str:
        return self._current_model_key

    @property
    def current_model_info(self) -> dict:
        return AVAILABLE_MODELS.get(self._current_model_key, AVAILABLE_MODELS[DEFAULT_MODEL_KEY])

    def get_class_names(self, model_key: Optional[str] = None) -> dict:
        """获取指定模型的类别名映射"""
        key = model_key or self._current_model_key
        info = AVAILABLE_MODELS.get(key, AVAILABLE_MODELS[DEFAULT_MODEL_KEY])
        return info.get("class_names", {})

    def switch_model(self, model_key: str) -> dict:
        """切换当前使用的模型，返回模型信息"""
        if model_key not in AVAILABLE_MODELS:
            model_key = DEFAULT_MODEL_KEY
        self._current_model_key = model_key
        # 确保模型已加载
        self._load_yolo(model_key)
        info = AVAILABLE_MODELS[model_key]
        print(f"🔄 已切换模型: {info['name']} ({model_key})")
        return info

    def get_available_models(self) -> list[dict]:
        """返回所有可用模型列表（供前端调用）"""
        models_list = []
        for key, info in AVAILABLE_MODELS.items():
            model_path = info["path"]
            models_list.append({
                "key": key,
                "name": info["name"],
                "description": info["description"],
                "species": list(info["class_names"].values()),
                "has_breed_classifier": info.get("has_breed_classifier", False),
                "for_monitor": info.get("for_monitor", False),
                "is_available": model_path.exists(),
                "file_name": model_path.name,
            })
        return models_list

    def _load_yolo(self, model_key: str) -> YOLO:
        """加载 YOLO 模型（带缓存）"""
        if model_key in self._models:
            return self._models[model_key]

        info = AVAILABLE_MODELS.get(model_key)
        if info is None:
            raise ValueError(f"未知模型: {model_key}")

        path = info["path"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if path.exists():
            model = YOLO(str(path))
            self._models[model_key] = model
            print(f"YOLO 模型已加载: {info['name']} ({path.name}, device={device})")
        else:
            print(f"⚠ 模型文件 {path} 不存在，使用 yolo11n.pt 预训练权重")
            model = YOLO("yolo11n.pt")
            self._models[model_key] = model
        return model

    def load_model(self, model_path: Optional[Path] = None) -> YOLO:
        """向后兼容：加载默认模型"""
        return self._load_yolo(self._current_model_key)

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

    def _should_classify_breed(self, class_name: str, model_key: Optional[str] = None) -> bool:
        """判断是否需要对检测结果做品种分类"""
        key = model_key or self._current_model_key
        info = AVAILABLE_MODELS.get(key, {})
        return (
            info.get("has_breed_classifier", False)
            and class_name == "荷兰猪"
            and self._classifier is not None
        )

    def detect(self, image: np.ndarray, conf_threshold: float = 0.25,
               model_key: Optional[str] = None) -> list[dict]:
        key = model_key or self._current_model_key
        model = self._load_yolo(key)
        class_names = self.get_class_names(key)
        results = model(image, conf=conf_threshold, verbose=False)

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    class_name = class_names.get(cls_id, f"未知({cls_id})")

                    det = {
                        "class_id": cls_id,
                        "class_name": class_name,
                        "confidence": round(conf, 4),
                        "bbox": [round(v, 1) for v in xyxy],
                        "breed_info": None,
                        "model_used": key,
                    }

                    # 如果是荷兰猪且模型支持品种分类，用分类器识别品种花色
                    if self._should_classify_breed(class_name, key):
                        det["breed_info"] = self._classifier.classify(image, xyxy)

                    detections.append(det)

        return detections

    def detect_image_file(self, image_path: Path, conf_threshold: float = 0.25,
                          model_key: Optional[str] = None) -> list[dict]:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        return self.detect(img, conf_threshold, model_key=model_key)

    @staticmethod
    def draw_boxes(image: np.ndarray, detections: list[dict]) -> np.ndarray:
        """绘制检测框和中文标签（使用 Pillow 渲染中文，OpenCV 默认字体不支持中文）"""
        from PIL import Image, ImageDraw, ImageFont

        # OpenCV BGR → PIL RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        # 尝试加载中文字体，找不到则用默认（至少不会乱码成???）
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",        # 微软雅黑
            "C:/Windows/Fonts/simhei.ttf",      # 黑体
            "C:/Windows/Fonts/simsun.ttc",      # 宋体
            "C:/Windows/Fonts/msyhbd.ttc",      # 微软雅黑粗体
        ]
        font = None
        for fp in font_paths:
            if Path(fp).exists():
                font = ImageFont.truetype(fp, 20)
                break
        if font is None:
            font = ImageFont.load_default()

        colors_rgb = {
            "荷兰猪": (0, 255, 0), "鹦鹉": (255, 0, 0), "仓鼠": (0, 0, 255),
            "奶黄加州": (0, 255, 128), "三花美泰": (128, 255, 0), "白加州": (200, 255, 200),
            "食盆": (255, 180, 0), "水壶": (0, 150, 255),
        }

        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            cls_name = det["class_name"]
            conf = det["confidence"]
            color = colors_rgb.get(cls_name, (255, 255, 255))

            # 画框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            # 构建标签
            if det.get("breed_info") and det["breed_info"].get("full_name") != "未知":
                bi = det["breed_info"]
                label = f"{bi.get('color','')}{bi.get('breed','')} ({conf:.0%})"
            else:
                label = f"{cls_name} {conf:.2f}"

            # 画标签背景 + 文字
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            label_y = max(y1 - th - 4, 0)
            draw.rectangle([x1, label_y, x1 + tw + 4, label_y + th + 4], fill=(*color, 180))
            text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
            draw.text((x1 + 2, label_y + 2), label, fill=text_color, font=font)

        # PIL RGB → OpenCV BGR
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


detector = YOLODetector()

