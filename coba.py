import cv2
import numpy as np
from ultralytics import YOLO
## KHUSUS YOLO KALIBRASI MANUAL DENGAN TOMBOL C


# Load model YOLOv8 (pretrained)
model = YOLO("yolov8n.pt")

# Kamera iVCam (biasanya camera laptop = 0, ivcam = 1)
cap = cv2.VideoCapture(1)

# === SET MANUAL ===
REAL_WIDTH_CM = 20.4  # lebar objek referensi (dalam satuan cm)
pixel_per_cm = None

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        width_pixel = x2 - x1
        height_pixel = y2 - y1

        # Kalibrasi (tekan 'c' saat objek referensi terlihat)
        if pixel_per_cm is None:
            label = "Calibrating..."
        else:
            length_cm = width_pixel / pixel_per_cm
            label = f"{length_cm:.2f} cm"

        # gambar bounding box
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("YOLO Measurement", frame)

    key = cv2.waitKey(1)

    # tekan 'c' untuk kalibrasi
    if key == ord('c'):
        # ambil bounding box pertama sebagai referensi
        if len(results.boxes) > 0:
            ref_box = results.boxes[0]
            x1, y1, x2, y2 = map(int, ref_box.xyxy[0])
            pixel_width = x2 - x1

            pixel_per_cm = pixel_width / REAL_WIDTH_CM
            print("Kalibrasi selesai:", pixel_per_cm, "pixel/cm")

    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()