# Manual reference width, Press r and type the known width (cm) of the object you will use for calibration.
# On‑screen display	Shows current reference width and scale (px/cm) directly on the video feed.
# Optional distance‑based mode	Press 'D' to enable. You enter the distance (cm) from camera to the reference object. The code then computes the camera’s focal length and can theoretically estimate size of objects at any distance (still approximate – requires consistent distance per object).
# Manual scale override	Press m to directly type a pixel_per_cm value – useful if you already know the correct scale from another method.
# Better feedback	Console prints the calibration results and any errors.

#tombol yang bisa dipencet:
#D -> input jarak dalam satuan cm
#M -> input pixel per cm
#R -> input ukuran dalam cm
#C -> Mulai (sudah dikalibrasi dengan ukuran dan jarak)

import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Camera (adjust index if needed)
cap = cv2.VideoCapture(1)

# === USER SETTINGS (can be changed interactively) ===
reference_width_cm = 16.4   # will be updated via 'r' key
pixel_per_cm = None

# For distance‑based scaling (optional, if you know the distance to reference)
camera_focal_length_px = None   # will be computed if you set distance
reference_distance_cm = 30     # distance from camera to reference object
use_distance_mode = False        # toggle with 'd'

def compute_pixel_per_cm_from_distance(real_width_cm, width_px, distance_cm, focal_px):
    """
    If you know the distance to the object, you can compute focal length and then
    scale for other objects at unknown distances. This is still approximate.
    """
    if focal_px is None:
        # Compute focal length from this reference
        focal_px = (width_px * distance_cm) / real_width_cm
        return focal_px, focal_px / distance_cm
    else:
        # Use existing focal length to compute pixel_per_cm at any distance
        pixel_per_cm_at_dist = focal_px / distance_cm
        return focal_px, pixel_per_cm_at_dist

print("=== YOLO Measurement with Manual Input ===")
print("Controls:")
print("  'c' : Calibrate using the first detected object (uses current reference width)")
print("  'r' : Set reference object width (cm) via console")
print("  'd' : Toggle distance mode & set reference distance (cm) - experimental")
print("  'm' : Manually enter pixel_per_cm value")
print("  ESC : Exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    
    # Draw instructions on frame
    cv2.putText(frame, f"Ref width: {reference_width_cm:.1f} cm", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    if pixel_per_cm is not None:
        cv2.putText(frame, f"Scale: {pixel_per_cm:.2f} px/cm", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    if use_distance_mode and reference_distance_cm:
        cv2.putText(frame, f"Ref distance: {reference_distance_cm:.0f} cm", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Process each detected object
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        width_px = x2 - x1
        height_px = y2 - y1

        # Determine label based on calibration state
        if pixel_per_cm is None:
            label = "Uncalibrated"
        else:
            # Simple width‑based length
            length_cm = width_px / pixel_per_cm
            label = f"{length_cm:.1f} cm (w)"
            # If you want height as well:
            # height_cm = height_px / pixel_per_cm
            # label = f"{length_cm:.1f}cm x {height_cm:.1f}cm"

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("YOLO Measurement", frame)
    key = cv2.waitKey(1) & 0xFF

    # --- CALIBRATE with current reference width ---
    if key == ord('c'):
        if len(results.boxes) > 0:
            ref_box = results.boxes[0]
            x1, y1, x2, y2 = map(int, ref_box.xyxy[0])
            pixel_width = x2 - x1
            if use_distance_mode and reference_distance_cm:
                # Distance‑based calibration (computes focal length)
                focal, scale = compute_pixel_per_cm_from_distance(
                    reference_width_cm, pixel_width, reference_distance_cm, camera_focal_length_px)
                # Update global focal and pixel_per_cm
                globals()['camera_focal_length_px'] = focal
                pixel_per_cm = scale
                print(f"Distance mode: focal length = {focal:.1f} px, scale = {pixel_per_cm:.3f} px/cm at ref distance")
            else:
                # Standard calibration
                pixel_per_cm = pixel_width / reference_width_cm
                print(f"Calibrated: {pixel_width} px / {reference_width_cm} cm = {pixel_per_cm:.3f} px/cm")
        else:
            print("No object detected. Can't calibrate.")

    # --- Set reference object width (cm) ---
    elif key == ord('r'):
        try:
            new_width = float(input("Enter reference object width (in cm): "))
            reference_width_cm = new_width
            print(f"Reference width set to {reference_width_cm} cm")
            # Optional: reset calibration because scale may change
            pixel_per_cm = None
            if use_distance_mode:
                globals()['camera_focal_length_px'] = None
        except ValueError:
            print("Invalid number. Keep previous width.")

    # --- Toggle distance mode and set reference distance ---
    elif key == ord('d'):
        use_distance_mode = not use_distance_mode
        if use_distance_mode:
            try:
                dist = float(input("Enter distance from camera to reference object (in cm): "))
                reference_distance_cm = dist
                print(f"Distance mode ON. Reference distance = {reference_distance_cm} cm")
                # Reset calibration because we need new focal length
                pixel_per_cm = None
                globals()['camera_focal_length_px'] = None
            except ValueError:
                print("Invalid distance. Distance mode disabled.")
                use_distance_mode = False
        else:
            print("Distance mode OFF. Using standard single‑plane calibration.")
            reference_distance_cm = None
            pixel_per_cm = None

    # --- Manually set pixel_per_cm (advanced override) ---
    elif key == ord('m'):
        try:
            manual_px_per_cm = float(input("Enter pixel_per_cm value manually: "))
            pixel_per_cm = manual_px_per_cm
            print(f"Manual scale set to {pixel_per_cm:.3f} px/cm")
        except ValueError:
            print("Invalid number. Scale unchanged.")

    elif key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()