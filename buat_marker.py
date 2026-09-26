import cv2

# 1. Pilih jenis ArUco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# 2. Tentukan ukuran gambar marker (semakin besar, semakin mudah dideteksi)
marker_size = 400

# 3. Generate marker dengan ID 0
marker_id = 0
marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

# 4. Simpan sebagai file gambar
cv2.imwrite(f"aruco_marker_{marker_id}.png", marker_img)
print(f"✅ Marker ArUco ID {marker_id} berhasil dibuat.")