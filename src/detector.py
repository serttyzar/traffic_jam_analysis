import numpy as np
from ultralytics import YOLO


class YoloDetector:
    def __init__(self, model_name='yolov8n.pt'):
        """Инициализация модели YOLO."""
        self.model = YOLO(model_name)
        self.dataset_classes = {
            2: 'car', 3: 'truck', 5: 'bus', 6: 'train'
        }
        self.class_ids = [2, 3, 5, 6]  # car, motorcycle, bus, train
        self.prev_tracks = {}

    def track(self, frame):
        """Трекинг транспорта на кадре с использованием ByteTrack."""
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        tracks = []
        speeds = []

        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                if cls_id in self.class_ids and box.conf[0] > 0.3:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    track_id = int(box.id[0]) if box.id is not None else -1
                    conf = float(box.conf[0])
                    # Вычисление скорости
                    curr_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    if track_id in self.prev_tracks:
                        prev_center = self.prev_tracks[track_id]
                        speed = np.sqrt((curr_center[0] - prev_center[0]) ** 2 + (curr_center[1] - prev_center[1]) ** 2)
                        speeds.append(speed)
                    self.prev_tracks[track_id] = curr_center
                    tracks.append((x1, y1, x2, y2, conf, cls_id, track_id))

        active_ids = {int(box.id[0]) for box in results[0].boxes if box.id is not None}
        self.prev_tracks = {k: v for k, v in self.prev_tracks.items() if k in active_ids}

        return tracks, speeds