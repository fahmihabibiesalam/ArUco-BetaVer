import cv2
import numpy as np
from ultralytics import YOLO

# === 1. PERSIAPAN ===
# Ukur panjang sisi marker asli dengan penggaris (contoh: 5 cm)
MARKER_UKURAN_ASLI_CM = 5.0  # <-- BISA DI-GANTI / DISESUAIKAN!

# Inisialisasi detektor ArUco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detektor_aruco = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Load model YOLO (pastikan file 'yolo11n.pt' ada di folder yang sama)
# Model ini sudah dilatih untuk mendeteksi 80 objek umum (orang, mobil, dll.)
model = YOLO('yolo11n.pt')

# Buka kamera (0 adalah ID untuk webcam default)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Tidak bisa membuka kamera.")
    exit()

print("✅ Kamera berhasil dibuka. Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- DETEKSI MARKER ArUco ---
    corners, ids, _ = detektor_aruco.detectMarkers(frame)
    skala_cm_per_piksel = None

    if ids is not None:
        # Gambar kotak di sekitar marker yang terdeteksi
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Ambil marker pertama yang terdeteksi
        marker_corners = corners[0][0]
        # Hitung lebar marker dalam piksel (rata-rata sisi atas dan bawah)
        lebar_piksel_atas = np.linalg.norm(marker_corners[0] - marker_corners[1])
        lebar_piksel_bawah = np.linalg.norm(marker_corners[2] - marker_corners[3])
        lebar_marker_piksel = (lebar_piksel_atas + lebar_piksel_bawah) / 2.0

        # HITUNG SKALA (cm per piksel)
        skala_cm_per_piksel = MARKER_UKURAN_ASLI_CM / lebar_marker_piksel
        # Tampilkan skala di layar
        cv2.putText(frame, f"Skala: {skala_cm_per_piksel:.4f} cm/px", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # --- DETEKSI BENDA DENGAN YOLO ---
    # YOLO akan mendeteksi benda-benda di dalam frame
    results = model(frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Dapatkan koordinat bounding box (kotak pembatas)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            lebar_benda_piksel = x2 - x1
            tinggi_benda_piksel = y2 - y1

            # Gambar bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{model.names[int(box.cls[0])]}: {box.conf[0]:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Jika skala ditemukan, hitung ukuran dalam cm
            if skala_cm_per_piksel is not None:
                lebar_cm = lebar_benda_piksel * skala_cm_per_piksel
                tinggi_cm = tinggi_benda_piksel * skala_cm_per_piksel
                ukuran_teks = f"Ukuran: {lebar_cm:.1f} x {tinggi_cm:.1f} cm"
                cv2.putText(frame, ukuran_teks, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Tampilkan hasil di layar
    cv2.imshow('Pengukuran Benda dengan YOLO', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan semua sumber daya
cap.release()
cv2.destroyAllWindows()