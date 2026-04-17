from ultralytics import YOLO


model = YOLO("yolov8n.pt")

model.train(
    data=r"C:/Jupyter/Object Tracking/datasets/maritime_detection/data.yaml",
    epochs=1,
    imgsz=606
)


