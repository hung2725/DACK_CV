# Hệ Thống Giám Sát Giao Thông và Phát Hiện Vi Phạm

Đây là hệ thống tự động nhận diện phương tiện giao thông và trạng thái đèn tín hiệu từ video để phát hiện lỗi vi phạm (vượt đèn đỏ), sử dụng mô hình học sâu YOLO.

## 1. Công nghệ sử dụng

- **Ngôn ngữ lập trình**: Python
- **Thư viện chính**:
  - `ultralytics` (YOLO): Sử dụng cho việc nhận diện đối tượng (Object Detection) và theo dõi đối tượng (Object Tracking). Hệ thống tải các file mô hình như `vehicle.pt` để nhận diện xe, và `traffic_light.pt` để nhận diện trạng thái đèn giao thông.
  - `opencv-python` (`cv2`): Sử dụng để xử lý video, đọc từng frame, vẽ khung bounding box, vẽ vạch kẻ và viết thông tin lên khung hình giao diện người dùng.
  - `numpy`: Hỗ trợ tính toán xử lý ma trận và hình ảnh bên dưới.

## 2. Cách hoạt động của hệ thống

Hệ thống hoạt động dựa trên luồng xử lý tuần tự qua từng frame hình ảnh của luồng video:

1. **Khởi tạo và cấu hình (`main.py`)**: 
   - Đọc đối tượng video qua OpenCV.
   - Khởi tạo đường kẻ ngang (vạch dừng ảo) thông qua tham số `stop_line_y` trong `ViolationChecker`. Hàm này sẽ quyết định vạch dừng thực tế trên video.

2. **Nhận diện và phân loại (`detection.py`)**:
   - Sử dụng định hình mô hình xe (`vehicle.pt`) kết hợp với chế độ tracking (`track`) của YOLO để nhận diện và theo dõi tọa độ của từng vị trí xe hơi/xe máy. Mỗi xe được gắn một ID duy nhất (`v_res.boxes.id`) để liên kết xuyên suốt các frame hình.
   - Mô hình phân loại tín hiệu đèn (`traffic_light.pt`) sẽ trích xuất ra trạng thái hiện tại của đèn tín hiệu (Ví dụ: `red`, `green`,... ).

3. **Kiểm tra điều kiện và xử lý vi phạm (`traffic_violation.py`)**:
   - **Xác định tín hiệu đèn**: Nếu chuỗi trạng thái đèn chứa `"red"` hoặc `"0"`, hệ thống xem như đèn đỏ và chuyển vạch dừng giới hạn sang màu đỏ.
   - **Xác định hành vi vượt vạch**: Thuật toán lưu trữ toạ độ `y2` (cạnh dưới cùng của box nhận diện chiếc xe) ở lần xuất hiện trước đó bằng biến `pre_position`.
   - Tiến hành đánh giá: Nếu đang là **đèn đỏ**, và xe từ frame trước đang ở phía ngoài vạch (`prev_y2 >= stop_line_y`), nhưng ở frame ngay sau đuôi xe đã ở qua vạch này định dạng (`y2 < stop_line_y`), thì hệ thống gán nhãn chiếc xe có ID này đã vi phạm lỗi vượt vạch đèn đỏ.
   - Hiển thị và thống kê: Xe bị đánh dấu vi phạm sẽ bị đóng khung màu đỏ, chữ `VI PHAM: [LOẠI XE]` chạy theo. Đồng thời bảng thống kê (góc trái trên màn hình) tự động tăng đếm số lượt vi phạm.

4. **Hiển thị trực quan**: Cửa sổ OpenCV sẽ chiếu video theo thời gian thực đè cùng nhận diện để trực quan diễn biến theo dõi.

## 3. Hướng dẫn chạy dự án

### Yêu cầu tiên quyết
Bạn cần chắc chắn đã cài đặt sẵn Python 3.

### Bước 1: Khởi tạo môi trường ảo (Khuyến nghị)
Sử dụng môi trường ảo để không bị xung đột phiên bản với các dự án khác:
```bash
python -m venv venv

# Kích hoạt trên Windows:
venv\Scripts\activate

# Kích hoạt trên macOS/Linux:
source venv/bin/activate
```

### Bước 2: Cài đặt các thư viện cần dùng
Sử dụng file `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Bước 3: Chuẩn bị mô hình (Trọng số YOLO)
Hệ thống mặc định yêu cầu thư mục `models/` phải có sẵn file trọng số. Nếu chưa có, bạn cần:
1. Tạo thư mục con tên là `models` ở đường dẫn chứa file `main.py`.
2. Đặt trọng số được train (model YOLO của bạn): `vehicle.pt`, `traffic_light.pt` vào trong thư mục `models/`.

### Bước 4: Chạy dự án
Đảm bảo bạn có file video (ví dụ `16h15.15.9.22.mp4`) để chạy thử ở thư mục hiện hành. Sau đó chạy lệnh:
```bash
python main.py
```

*Lưu ý khi sử dụng*: 
- Trọng số và nguồn video tuỳ chỉnh có thể thay đổi bằng việc cập nhật source code ở khai báo nguồn `cv2.VideoCapture("...")` bên trong `main.py`.
- Tuỳ chỉnh toạ độ vạch dừng thẳng đứng (`stop_line_y=380`) bằng cách thay đổi giá trị trong lần gọi `ViolationChecker(stop_line_y=380)` để khớp nhất với phối cảnh video của riêng bạn.
- Bạn có thể nhấn phím `q` tại giao diện video đang chạy bất cứ lúc nào để thoát ứng dụng khẩn cấp.
