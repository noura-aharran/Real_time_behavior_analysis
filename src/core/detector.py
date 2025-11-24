# src/core/detector.py
import cv2
import numpy as np

# Try to use ultralytics YOLO (yolov8). If not available, fallback to OpenCV HOG person detector.
try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    _HAS_YOLO = False

class Detection:
    def __init__(self, x1, y1, x2, y2, conf=1.0, cls='person'):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        self.conf = float(conf)
        self.cls = cls

    def to_xyxy(self):
        return self.x1, self.y1, self.x2, self.y2

class Detector:
    def __init__(self, model_path=None, conf_thres=0.35):
        self.conf_thres = conf_thres
        if _HAS_YOLO and model_path is not None:
            try:
                self.model = YOLO(model_path)
            except Exception:
                self.model = YOLO("yolov8n.pt")
        elif _HAS_YOLO:
            self.model = YOLO("yolov8n.pt")
        else:
            # OpenCV HOG fallback
            self.model = None
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame):
        """
        Retourne une liste de Detection.
        """
        if _HAS_YOLO and self.model is not None:
            # ultralytics model expects RGB
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(img, imgsz=640, conf=self.conf_thres, verbose=False)
            # results is a list with one item per image
            dets = []
            if len(results) > 0:
                r = results[0]
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls.cpu().numpy()[0]) if hasattr(box, 'cls') else None
                    conf = float(box.conf.cpu().numpy()[0]) if hasattr(box, 'conf') else 1.0
                    xyxy = box.xyxy.cpu().numpy()[0]
                    x1,y1,x2,y2 = xyxy
                    # only keep person class (COCO id 0) if class info exists
                    if cls_id is None or cls_id == 0:
                        dets.append(Detection(x1,y1,x2,y2, conf=conf, cls='person'))
            return dets
        else:
            # HOG detector: returns rects as (x, y, w, h)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects, weights = self.hog.detectMultiScale(gray, winStride=(8,8), padding=(8,8), scale=1.05)
            dets = []
            for (x, y, w, h), conf in zip(rects, weights):
                x1,y1,x2,y2 = x, y, x+w, y+h
                dets.append(Detection(x1,y1,x2,y2, conf=float(conf), cls='person'))
            return dets
