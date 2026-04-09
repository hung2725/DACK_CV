import torch
from ultralytics import YOLO

def start_train():
    torch.cuda.empty_cache()
    tasks = [
        {"yaml": "data/vehicle_data/data.yaml", "name": "vehicle_model", "img": 640},
        {"yaml": "data/traffic_light_data/data.yaml", "name": "light_model", "img": 416},
    ]

    for task in tasks:
        print(f"\nĐANG HUẤN LUYỆN: {task['name'].upper()}")
        model = YOLO("yolo26n.pt") 
        
        model.train(
            data=task['yaml'],
            epochs=150,
            imgsz=task['img'],
            batch=8,
            workers=0,
            device=0,
            name=task['name'],
            amp=True,
            exist_ok=True
        )
        torch.cuda.empty_cache()

if __name__ == "__main__":
    start_train()