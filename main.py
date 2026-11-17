
import cv2
import time
import os
from ultralytics import YOLO

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

# Load pretrained YOLOv8 model
model = YOLO("yolov8s.pt")

# Video source
# 0 = Webcam
# "video.mp4" = Video file
SOURCE = 0

# Confidence threshold
CONFIDENCE = 0.4

# Create output directory
os.makedirs("output", exist_ok=True)

# ==========================================
# 2. OPEN VIDEO SOURCE
# ==========================================

cap = cv2.VideoCapture(SOURCE)

if not cap.isOpened():
    print("Error: Unable to open webcam/video!")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps_video = cap.get(cv2.CAP_PROP_FPS)

if fps_video <= 0:
    fps_video = 30

# ==========================================
# 3. VIDEO OUTPUT CONFIGURATION
# ==========================================

out = cv2.VideoWriter(
    "output/tracked_video.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps_video,
    (width, height)
)

# ==========================================
# 4. OBJECT DETECTION AND TRACKING
# ==========================================

prev_time = time.time()

print("Object Detection and Tracking Started!")
print("Press Q to exit.")

while True:

    # Read video frame
    ret, frame = cap.read()

    if not ret:
        print("Video ended or frame unavailable.")
        break

    # YOLO object detection + ByteTrack
    results = model.track(
    source=frame,
    persist=True,
    tracker="bytetrack.yaml",
    conf=0.25,
    iou=0.5,
    imgsz=960,
    verbose=False
)

    result = results[0]

    # ======================================
    # 5. DRAW BOUNDING BOXES AND TRACK IDS
    # ======================================

    if result.boxes is not None and len(result.boxes) > 0:

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        if result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()
        else:
            track_ids = [None] * len(boxes)

        for box, cls, conf, track_id in zip(
            boxes, classes, confidences, track_ids
        ):

            x1, y1, x2, y2 = map(int, box)

            class_name = model.names[int(cls)]

            # Create object label
            if track_id is not None:
                label = f"{class_name} ID:{track_id} {conf:.2f}"
            else:
                label = f"{class_name} {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Draw label background
            cv2.rectangle(
                frame,
                (x1, max(y1 - 30, 0)),
                (min(x1 + 220, width), y1),
                (0, 255, 0),
                -1
            )

            # Display object name and tracking ID
            cv2.putText(
                frame,
                label,
                (x1 + 5, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

    # ======================================
    # 6. CALCULATE FPS
    # ======================================

    current_time = time.time()

    fps = 1 / max(current_time - prev_time, 0.000001)

    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # ======================================
    # 7. DISPLAY AND SAVE OUTPUT
    # ======================================

    out.write(frame)

    cv2.imshow(
        "YOLOv8 Object Detection and Tracking",
        frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================================
# 8. RELEASE RESOURCES
# ==========================================

cap.release()
out.release()

cv2.destroyAllWindows()

print("Tracking completed successfully!")
print("Output saved: output/tracked_video.mp4")
