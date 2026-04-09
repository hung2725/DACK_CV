import cv2
import time # Thêm thư viện time để đo độ trễ
from detection import TrafficDetector
from traffic_violation import ViolationChecker

def run_system():
    detector = TrafficDetector()
    checker = ViolationChecker(stop_line_y=350) 
    
    cap = cv2.VideoCapture("16h15.15.9.22.mp4")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    output_filename = "ket_qua_vi_pham_co_FPS.mp4"
    out = cv2.VideoWriter(output_filename, fourcc, video_fps, (frame_width, frame_height))
    print(f"Đang xử lý và lưu video ra file: {output_filename} ...")

    # Các biến để làm mượt chỉ số (tránh số nhảy quá nhanh gây chóng mặt)
    avg_fps = 0
    avg_latency_ms = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # --- BẮT ĐẦU ĐO THỜI GIAN ---
        start_time = time.time()

        # Xử lý nhận diện và tracking
        detections, lights = detector.detect_all(frame)
        # Xử lý vi phạm và vẽ giao diện
        result_frame = checker.process(frame, detections, lights)

        # --- KẾT THÚC ĐO THỜI GIAN ---
        end_time = time.time()
        
        # Tính toán độ trễ (Latency) và FPS thực tế
        latency = end_time - start_time
        current_fps = 1 / latency if latency > 0 else 0
        current_latency_ms = latency * 1000

        # Làm mượt các con số bằng trung bình cộng có trọng số
        if avg_fps == 0:
            avg_fps = current_fps
            avg_latency_ms = current_latency_ms
        else:
            avg_fps = 0.9 * avg_fps + 0.1 * current_fps
            avg_latency_ms = 0.9 * avg_latency_ms + 0.1 * current_latency_ms
 
        # --- VẼ CHỈ SỐ LÊN GÓC PHẢI MÀN HÌNH ---
        # Vẽ khung nền đen mờ nhỏ để chữ dễ đọc hơn
        cv2.rectangle(result_frame, (frame_width - 280, 10), (frame_width - 20, 80), (30, 30, 30), -1)
        
        # Ghi chữ FPS
        cv2.putText(result_frame, f"FPS: {int(avg_fps)}", (frame_width - 260, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Ghi chữ Latency (ms)
        cv2.putText(result_frame, f"Latency: {int(avg_latency_ms)}ms", (frame_width - 260, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Ghi frame đã xử lý vào file video mới
        out.write(result_frame)

        cv2.imshow("Giam Sat Giao Thong - YOLO 2026", result_frame)
        if cv2.waitKey(1) & 0xFF == ord(' '): 
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Đã lưu video thành công!")

if __name__ == "__main__":
    run_system()