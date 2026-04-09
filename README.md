# Dự Án Nhận Diện Giao Thông Bằng YOLO (Phát Hiện Vượt Đèn Đỏ)

Dự án này là một hệ thống AI nhỏ gọn giúp tự động phát hiện xe nào vượt đèn đỏ bằng Camera. Thay vì người thật phải ngồi căng mắt xem video, hệ thống sẽ làm giúp điều đó.

Dưới đây là lời giải thích **đơn giản và dễ hiểu nhất** về cách từng file code hoạt động trong dự án của bạn!

---

## 1. File `detection.py` (Con mắt của hệ thống)

Nhiệm vụ của file này là "nhìn" vào màn hình và trả lời 2 câu hỏi:
- Có những xe nào đang chạy?
- Đèn giao thông đang sáng màu gì?

```python
import cv2
import os
from ultralytics import YOLO

class TrafficDetector:
    def __init__(self):
        # Tải 2 thư viện AI vào não: một cái dùng để nhìn xe, một cái dùng để nhìn đèn
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.vehicle_model = YOLO(os.path.join(BASE_DIR, 'models', 'vehicle.pt'))
        self.light_model = YOLO(os.path.join(BASE_DIR, 'models', 'traffic_light.pt'))

    def detect_all(self, frame):
        # 1. TÌM XE CỘ
        # Dùng .track() để AI biết "xe này ở khung hình trước chính là xe này ở khung hình hiện tại"
        # Giống như bạn dán số báo danh cho từng chiếc xe vậy.
        v_res = self.vehicle_model.track(frame, persist=True, imgsz=480, verbose=False)[0]
        
        detections = []
        if v_res.boxes.id is not None:
            # Lấy thông tin xe đã được dán số báo danh
            ids = v_res.boxes.id.int().cpu().tolist()      # Mã số của xe (ví dụ: Xe 1, Xe 2)
            bboxes = v_res.boxes.xyxy.int().cpu().tolist() # Vị trí khung hình vuông bao quanh xe
            clss = v_res.boxes.cls.int().cpu().tolist()    # Đây là xe ô tô hay xe máy?
            
            # Lưu lại vào danh sách
            for box, id, cls in zip(bboxes, ids, clss):
                detections.append({
                    "box": box, 
                    "id": id, 
                    "type": self.vehicle_model.names[cls] 
                })

        # 2. TÌM ĐÈN GIAO THÔNG
        # Đèn thì đứng yên nên ta chỉ cần tìm (.predict) chứ không cần theo dõi số báo danh (.track)
        l_res = self.light_model(frame, imgsz=320, verbose=False)[0]
        lights = []
        for l_box in l_res.boxes:
            lx1, ly1, lx2, ly2 = map(int, l_box.xyxy[0])  
            status = self.light_model.names[int(l_box.cls[0])] # Đèn đang báo chữ gì (Red, Green, Yellow)
            lights.append({"box": [lx1, ly1, lx2, ly2], "status": status})
        
        # Đưa danh sách xe và danh sách đèn ra ngoài để xử lý tiếp
        return detections, lights
```

---

## 2. File `traffic_violation.py` (Cảnh sát bắt lỗi)

File này sẽ cầm danh sách các xe và trạng thái đèn ở trên để xem ai phạm luật.

**Nguyên lý bắt lỗi đơn giản:**
- Camera quay từ trên cao xuống.
- Vạch dừng xe nằm ở giữa màn hình (Tọa độ `Y = 350`).
- Xe chạy từ dưới đáy màn hình (Y lớn) tiến dần lên phía trên đỉnh màn hình (Y nhỏ dần).
- **Phạm luật khi:** Đèn đang đỏ + Lúc nãy mông chiếc xe nằm dưới vạch (Y >= 350) + Tự nhiên chớp mắt cái xe đã vụt nằm bên trên vạch (Y < 350).

```python
import cv2

class ViolationChecker:
    def __init__(self, stop_line_y=350):
        self.stop_line_y = stop_line_y  # Cắm một cái vạch ảo ở tọa độ 350
        
        self.stats = {}            # Bảng ghi sổ bao nhiêu xe vi phạm
        self.violated_ids = set()  # Sổ đen ghi số báo danh của xe vi phạm, để lỡ nó vẫn đang chạy thì không bị bắt phạt 2 lần
        self.pre_position = {}     # Sổ ghi chép vị trí cũ của xe ở khung hình liền trước đó

    def process(self, frame, detections, lights):
        # 1. KIỂM TRA ĐÈN
        is_red = False 
        for l in lights:
            lx1, ly1, lx2, ly2 = l['box']
            status = str(l['status']).lower()
            
            # Chỉ cần AI đọc chữ có 'red', 'do' hoặc là '0' thì hiểu là Đèn Đỏ.
            l_color = (0, 0, 255) if any(x in status for x in ['red', 'do', '0']) else (0, 255, 0)
            if l_color == (0, 0, 255): 
                is_red = True # Kích hoạt chế độ đi săn
            
            # Vẽ hình cái đèn lên màn hình cho dễ nhìn
            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), l_color, 2)
            cv2.putText(frame, status.upper(), (lx1, ly1 - 10), 0, 0.6, l_color, 2)

        # 2. VẼ VẠCH KẺ
        line_color = (0, 0, 255) if is_red else (0, 255, 0)
        cv2.line(frame, (0, self.stop_line_y), (frame.shape[1], self.stop_line_y), line_color, 3)
        
        # ...Đoạn này vẽ cái khung nền đen cho bảng thống kê (mình ẩn cho gọn code)...

        # 3. TIẾN HÀNH BẮT LỖI
        for v in detections:
            x1, y1, x2, y2 = v['box']
            v_id, v_type = v['id'], v['type']

            # Lấy vị trí đuôi xe (y2) ở 0.03 giây trước. Nếu xe mới xuất hiện thì coi như chưa có gì.
            prev_y2 = self.pre_position.get(v_id, y2)
            
            # LOGIC CHỐT ĐƠN:
            # - is_red: Đèn màu đỏ!
            # - prev_y2 >= 350: Lần trước nhìn thấy, đuôi xe vẫn đang nằm trước vạch an toàn.
            # - y2 < 350: Tự nhiên bây giờ đuôi xe đã nằm thò qua vạch!!
            if is_red and prev_y2 >= self.stop_line_y and y2 < self.stop_line_y:
                if v_id not in self.violated_ids:
                    # Ghi sổ lại xe này đã vi phạm!
                    self.stats[v_type] = self.stats.get(v_type, 0) + 1
                    self.violated_ids.add(v_id)

            # Vẽ chữ "VI PHAM" to tướng màu đỏ vào xe tội phạm
            if v_id in self.violated_ids:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(frame, f"VI PHAM: {v_type.upper()}", (x1, y1 - 20), 0, 0.7, (0, 0, 255), 2)
            else:
                # Xe hiền lành thì cứ vẽ màu xanh dương tính
                cv2.putText(frame, f"ID:{v_id} {v_type.upper()}", (x1, y1 - 10), 0, 0.5, (0, 255, 0), 1)

            # LƯU LẠI VỊ TRÍ để tí nữa lặp vòng tiếp theo thì nó biến thành quá khứ!
            self.pre_position[v_id] = y2

        # 4. In thành tích vi phạm lên góc màn hình cho đẹp
        y_pos = 100
        for vt, count in self.stats.items():
            cv2.putText(frame, f"- {vt.upper()}: {count} xe", (50, y_pos), 0, 0.6, (255, 255, 255), 1)
            y_pos += 30

        return frame
```

---

## 3. File `main.py` (Kịch bản ghép nối mọi thứ)

Một hệ thống hoàn chỉnh phải kết nối "Mắt" và "Cảnh sát" lại với nhau rồi mở Video lên chạy.

```python
import cv2
import time 
from detection import TrafficDetector
from traffic_violation import ViolationChecker

def run_system():
    # Gọi 2 ông bạn ở file trên ra làm nhiệm vụ
    detector = TrafficDetector()
    checker = ViolationChecker(stop_line_y=350) 
    
    # Mở Video lên xem
    cap = cv2.VideoCapture("16h15.15.9.22.mp4")
    
    # Khởi tạo công cụ để lát nữa lưu lại thành video kết quả MP4
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    output_filename = "ket_qua_vi_pham_co_FPS.mp4"
    out = cv2.VideoWriter(output_filename, fourcc, video_fps, (frame_width, frame_height))

    avg_fps = 0
    avg_latency_ms = 0

    # Lặp qua từng bức ảnh tĩnh trong đoạn Video (mỗi bức là 1 frame)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        start_time = time.time()  # Bắt đầu tính giờ

        # QUAN TRỌNG NHẤT:
        # Nhờ bạn "Mắt" AI khoanh vùng các xe và đèn từ trong bức ảnh của video đang chiếu
        detections, lights = detector.detect_all(frame)
        
        # Đưa thông tin cho bạn "Cảnh Sát" xử lý để xem có xe nào vượt vạch không
        result_frame = checker.process(frame, detections, lights)

        end_time = time.time()  # Xử lý xong 1 bức ảnh rùi nè
        
        # ĐO ĐẠC TỐC ĐỘ:
        # Ở đây chỉ là các phép toán tính xem 1 giây lướt được bao nhiêu bức ảnh (FPS)
        # Bọn mình dùng trung bình cộng để số FPS nó đỡ bị nhảy loạn xạ trên màn hình.
        latency = end_time - start_time
        current_fps = 1 / latency if latency > 0 else 0
        current_latency_ms = latency * 1000

        if avg_fps == 0:
            avg_fps = current_fps
            avg_latency_ms = current_latency_ms
        else:
            avg_fps = 0.9 * avg_fps + 0.1 * current_fps
            avg_latency_ms = 0.9 * avg_latency_ms + 0.1 * current_latency_ms

        # Viết số khung hình FPS ra góc màn hình 
        cv2.putText(result_frame, f"FPS: {int(avg_fps)}", (frame_width - 260, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result_frame, f"Latency: {int(avg_latency_ms)}ms", (frame_width - 260, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # XUẤT RA NGOÀI:
        out.write(result_frame) # Gắn bức ảnh vừa vẽ thêm chữ vào File Clip mới
        cv2.imshow("Giam Sat Giao Thong - YOLO 2026", result_frame) # Cứ mở rộng cửa sổ lên cho người dùng xem
        
        # Ai bấm nút Space trên bàn phím thì tắt chương trình.
        if cv2.waitKey(1) & 0xFF == ord(' '): 
            break

    # Dọn dẹp nhà cửa khi chạy xong
    cap.release()
    out.release()
    cv2.destroyAllWindows()
```

---

## 4. Cách để anh em tự chạy dự án này

**Bước 1:** Chuẩn bị môi trường (cài đặt mấy đồ linh tinh cần thiết)
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Bước 2:** Bỏ 2 cục "Não AI" vào đúng thư mục
Nhớ tạo thư mục `models/` ở cạnh file `main.py`, rồi copy 2 file `vehicle.pt` và `traffic_light.pt` vào đó nhé. Hệ thống không có 2 file này là mù đường đấy!

**Bước 3:** Chạy thử nào
Chỉ cần chạy lệnh dưới, sẽ tự hiện một giao diện bật video lên cho bạn xem nó khoanh đỏ xe nào!
```bash
python main.py
```
Muốn thoát giữa chừng chỉ cần ấn phim `Space` (Dấu cách). Video sau khi chiếu xong sẽ tự gói lại đàng hoàng trong tệp `ket_qua_vi_pham_co_FPS.mp4`.
