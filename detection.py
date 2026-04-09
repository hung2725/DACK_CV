import cv2
import os
from ultralytics import YOLO

class TrafficDetector:
    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.vehicle_model = YOLO(os.path.join(BASE_DIR, 'models', 'vehicle.pt'))
        self.light_model = YOLO(os.path.join(BASE_DIR, 'models', 'traffic_light.pt'))
        self.tracker_config = os.path.join(BASE_DIR, 'bytetrack_custom.yaml')

        # Remap ID: ánh xạ raw YOLO ID → ID nhỏ gọn, liên tục
        # Tránh trường hợp ID nhảy vọt từ vài trăm lên vài nghìn
        self._id_map = {}       # {raw_yolo_id: compact_id}
        self._next_id = 1       # Counter ID nhỏ cấp phát

    def _remap_id(self, raw_id: int) -> int:
        """Chuyển đổi YOLO raw ID (có thể rất lớn) về ID nhỏ gọn liên tục."""
        if raw_id not in self._id_map:
            self._id_map[raw_id] = self._next_id
            self._next_id += 1
        return self._id_map[raw_id]

    def detect_all(self, frame):
        # Tracking xe với custom config để giữ ID ổn định hơn
        v_res = self.vehicle_model.track(
            frame,
            persist=True,
            imgsz=480,
            verbose=False,
            tracker=self.tracker_config,
        )[0]

        detections = []
        if v_res.boxes.id is not None:
            ids   = v_res.boxes.id.int().cpu().tolist()
            bboxes = v_res.boxes.xyxy.int().cpu().tolist()
            clss  = v_res.boxes.cls.int().cpu().tolist()
            for box, raw_id, cls in zip(bboxes, ids, clss):
                compact_id = self._remap_id(raw_id)   # Dùng ID nhỏ gọn
                detections.append({
                    "box": box,
                    "id": compact_id,
                    "type": self.vehicle_model.names[cls],
                })

        # Nhận diện cột đèn và đóng khung
        l_res = self.light_model(frame, imgsz=320, verbose=False)[0]
        lights = []
        for l_box in l_res.boxes:
            lx1, ly1, lx2, ly2 = map(int, l_box.xyxy[0])
            status = self.light_model.names[int(l_box.cls[0])]
            lights.append({"box": [lx1, ly1, lx2, ly2], "status": status})

        return detections, lights