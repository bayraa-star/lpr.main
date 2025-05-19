import cv2
import torch
import numpy as np
from fast_alpr.base import BaseDetector, DetectionResult, BoundingBox

class YOLOv5Detector(BaseDetector):
    def __init__(self, model_path: str):
        """
        Initialize the YOLOv5 detector with your trained model.
        
        Args:
            model_path (str): Path to the trained YOLOv5 model (e.g., 'runs/train/exp3/weights/best.pt').
        """
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        self.class_names = self.model.names
    
    def predict(self, frame: np.ndarray) -> list[DetectionResult]:
        """
        Detect license plates in the input frame using the YOLOv5 model.
        
        Args:
            frame (np.ndarray): Input image in BGR format (from OpenCV).
        
        Returns:
            list[DetectionResult]: List of detected license plates with bounding boxes and confidence scores.
        """
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.model(img)

        detections = []
        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = det
            bbox = BoundingBox(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2))
            label = self.class_names[int(cls)]
            detections.append(DetectionResult(bounding_box=bbox, confidence=float(conf), label=label))
        
        return detections