import cv2
import numpy as np
import csv
import os
import time
from datetime import datetime
from threading import Lock

_last_log_time = 0.0
LOG_INTERVAL_SEC = 1.0
csv_lock = Lock()

def init_csv_logger():
    os.makedirs("Data", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"Data/Object Measurements_{timestamp}.csv"
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp", "Frame Index", "Object Label", "Marker Status",
            "cm/pixel", "Width (cm)", "Height (cm)", "Volume (cm^3)", "Margin of Error (%)"
        ])
    return filename

def log_measurement(filename, frame_idx, label, marker_status, cm_per_pixel, w_cm, h_cm, volume_cm3, err_total):
    global _last_log_time
    current_time = time.time()
    if current_time - _last_log_time < LOG_INTERVAL_SEC:
        return
    with csv_lock:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(filename, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, frame_idx, label, marker_status,
                    round(cm_per_pixel, 6) if cm_per_pixel else None,
                    round(w_cm, 3), round(h_cm, 3),
                    round(volume_cm3, 3), round(err_total, 3),
                ])
            _last_log_time = current_time
        except Exception as e:
            print(f"⚠️ Failed to write log: {e}")

def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, obj_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobel_x, sobel_y)
    sobel_mag = cv2.convertScaleAbs(sobel_mag)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(sobel_mag, kernel, iterations=1)
    overlay = cv2.addWeighted(frame, 0.8, cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR), 0.6, 0)
    return overlay, gray, blurred, obj_mask, sobel_mag, dilated

class DetectorObj:
    def detect_objects(self, frame):
        overlay, *_ = preprocess_image(frame)
        gray_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
        mask = cv2.adaptiveThreshold(gray_overlay, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, 19, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cnt for cnt in contours if cv2.contourArea(cnt) > 2000], overlay, mask

def compute_volume_and_error(contour, cm_per_pixel, true_dims=(17.5, 6.5)):
    x, y, w, h = cv2.boundingRect(contour)
    bun_mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(bun_mask, [contour - [x, y]], -1, 255, -1)
    diameters_px = []
    for col in range(w):
        col_pix = np.where(bun_mask[:, col] > 0)[0]
        if len(col_pix) > 1:
            diameters_px.append(col_pix.max() - col_pix.min())
    if len(diameters_px) <= 10:
        return None
    diameters_px = np.array(diameters_px)
    diameters_cm = diameters_px * cm_per_pixel
    delta_x_cm = cm_per_pixel
    volume_cm3 = np.sum(np.pi * (diameters_cm / 2) ** 2 * delta_x_cm)
    w_cm = w * cm_per_pixel
    h_cm = h * cm_per_pixel
    true_w, true_h = true_dims
    width_diff_pct = abs((w_cm - true_w) / true_w) * 10
    height_diff_pct = abs((h_cm - true_h) / true_h) * 10
    err_total = np.sqrt(0.7 * (width_diff_pct**2) + 0.3 * (height_diff_pct**2))
    return w_cm, h_cm, volume_cm3, err_total, (x, y, w, h)
