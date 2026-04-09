import os
from ultralytics import YOLO

def test_accuracy_percentage():
    # 1. Khai báo đường dẫn
    # Sửa lại "models/vehicle.pt" thành đường dẫn tới file trọng số của bạn nếu cần
    model_path = "models/vehicle.pt" 
    
    # BẮT BUỘC PHẢI CÓ FILE YAML VÀ THƯ MỤC ẢNH ĐỂ MÁY CÓ CÁI MÀ CHẤM ĐIỂM
    yaml_path = "data/vehicle_data/data.yaml" 

    print("="*60)
    print("BẮT ĐẦU KIỂM THỬ ĐỘ CHÍNH XÁC CỦA MÔ HÌNH YOLOV26")
    print("="*60)

    # 2. Kiểm tra xem file YAML có tồn tại không (để tránh cái lỗi FileNotFoundError lúc nãy)
    if not os.path.exists(yaml_path):
        print(f"LỖI: Không tìm thấy file {yaml_path}!")
        print("=> Máy tính không có ảnh gốc để test. Bạn phải tải lại Dataset từ Roboflow")
        print("=> và bỏ vào thư mục 'data/vehicle_data/' thì code này mới chấm điểm được nhé!")
        return

    try:
        # 3. Nạp mô hình
        print(f"\nĐang nạp mô hình từ: {model_path}...")
        model = YOLO(model_path)

        # 4. Chạy kiểm thử trên tập test
        print("\nĐang chạy test trên tập dữ liệu... (sẽ mất vài phút tùy máy)")
        # LƯU Ý: Nếu dataset của bạn không có thư mục 'test', hãy đổi split='test' thành split='val'
        metrics = model.val(data=yaml_path, split='test', imgsz=640)

        # 5. Rút trích số liệu và đổi sang Phần Trăm (%)
        map50 = metrics.box.map50 * 100
        precision = metrics.box.mp * 100
        recall = metrics.box.mr * 100

        # 6. In kết quả đẹp mắt ra màn hình
        print("\n" + "="*60)
        print("KẾT QUẢ ĐỘ CHÍNH XÁC TRÊN TẬP TEST (ĐÃ QUY ĐỔI RA %):")
        print("="*60)
        print(f"-> Độ chính xác trung bình (mAP@50) : {map50:.2f}%")
        print(f"-> Độ chuẩn xác (Precision)        : {precision:.2f}%")
        print(f"-> Độ phủ (Recall)                 : {recall:.2f}%")
        print("="*60)

    except Exception as e:
        print(f"\nĐã xảy ra lỗi trong quá trình test: {e}")

if __name__ == "__main__":
    test_accuracy_percentage()