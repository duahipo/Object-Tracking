import cv2
import os
from ultralytics import YOLO

model = YOLO("weights/yolov8_boat.pt")
video_path = "videos/boat_test.mp4"

output_dir = "datasets/maritime_reid/train"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_id = 0
boat_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4)

    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            boat_folder = f"{output_dir}/boat_{boat_id:03d}"
            os.makedirs(boat_folder, exist_ok=True)

            cv2.imwrite(
                f"{boat_folder}/frame_{frame_id}.jpg",
                crop
            )

            boat_id += 1

    frame_id += 1

cap.release()
print("✅ ReID dataset generated")
