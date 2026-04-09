import cv2

class ViolationChecker:
    def __init__(self, stop_line_y=350):
        self.stop_line_y = stop_line_y
        self.stats = {}            
        self.violated_ids = set()  
        self.pre_position = {}     # Lưu tọa độ y của xe ở frame trước đó

    def process(self, frame, detections, lights):
        # 1. Hiển thị cột đèn
        is_red = False
        for l in lights:
            lx1, ly1, lx2, ly2 = l['box']
            status = str(l['status']).lower()
            l_color = (0, 0, 255) if any(x in status for x in ['red', 'do', '0']) else (0, 255, 0)
            if l_color == (0, 0, 255): 
                is_red = True
            
            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), l_color, 2)
            cv2.putText(frame, status.upper(), (lx1, ly1 - 10), 0, 0.6, l_color, 2)

        # 2. Vẽ vạch và Bảng thống kê
        line_color = (0, 0, 255) if is_red else (0, 255, 0)
        cv2.line(frame, (0, self.stop_line_y), (frame.shape[1], self.stop_line_y), line_color, 3)
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (350, 180), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, "THONG KE VI PHAM:", (40, 60), 0, 0.7, (0, 255, 255), 2)

        # 3. Logic bắt lỗi VƯỢT/ĐÈ vạch
        for v in detections:
            x1, y1, x2, y2 = v['box']
            v_id, v_type = v['id'], v['type']

            # Kiểm tra xem frame trước xe này ở đâu
            prev_y2 = self.pre_position.get(v_id, y2)
            
            # ĐIỀU KIỆN VI PHAM:
            # - Đèn đang đỏ
            # - Frame trước: xe ở DƯỚI vạch (prev_y2 >= stop_line_y)
            # - Frame này: xe đã ĐÈ/VƯỢT lên vạch (y2 < stop_line_y)
            if is_red and prev_y2 >= self.stop_line_y and y2 < self.stop_line_y:
                if v_id not in self.violated_ids:
                    self.stats[v_type] = self.stats.get(v_type, 0) + 1
                    self.violated_ids.add(v_id)

            # Nếu đã xác định vi phạm, giữ khung đỏ cho đến khi xe đi khuất
            if v_id in self.violated_ids:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(frame, f"VI PHAM: {v_type.upper()}", (x1, y1 - 20), 0, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"ID:{v_id} {v_type.upper()}", (x1, y1 - 10), 0, 0.5, (0, 255, 0), 1)

            # Cập nhật vị trí để so sánh cho frame sau
            self.pre_position[v_id] = y2

        # Cập nhật bảng thống kê chữ trắng rõ ràng
        y_pos = 100
        for vt, count in self.stats.items():
            cv2.putText(frame, f"- {vt.upper()}: {count} xe", (50, y_pos), 0, 0.6, (255, 255, 255), 1)
            y_pos += 30

        return frame