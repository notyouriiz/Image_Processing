from flask import Flask, render_template, Response, send_from_directory, jsonify, request
import cv2, os, threading, numpy as np
from datetime import datetime
from measurement import DetectorObj, init_csv_logger, log_measurement, compute_volume_and_error, preprocess_image

app = Flask(__name__)

# Global Variables
cap = cv2.VideoCapture(0)
detector_obj = DetectorObj()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

marker_id_expected = 0
marker_size_cm = 5.0
cm_per_pixel = None
frame_idx = 0
running = True
selected_layer = "original"

csv_filename = init_csv_logger()
recent_records = []
lock = threading.Lock()

# Frame Generator
def gen_frames():
    global frame_idx, cm_per_pixel, running, selected_layer, recent_records
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not running:
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            continue

        frame_idx += 1
        img = frame.copy()

        # ArUco detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco_detector.detectMarkers(gray)
        marker_status = "No marker"
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == marker_id_expected:
                    cv2.polylines(img, [np.intp(corners[i][0])], True, (255, 0, 0), 2)
                    marker_status = f"Marker OK (ID {marker_id})"
                    side_px = np.median([
                        np.linalg.norm(corners[i][0][0] - corners[i][0][1]),
                        np.linalg.norm(corners[i][0][1] - corners[i][0][2]),
                        np.linalg.norm(corners[i][0][2] - corners[i][0][3]),
                        np.linalg.norm(corners[i][0][3] - corners[i][0][0]),
                    ])
                    if side_px > 10:
                        cm_per_pixel = marker_size_cm / side_px
                    break

        # Preprocessing
        overlay, gray_img, blurred, obj_mask, sobel_mag, dilated = preprocess_image(img)
        view_layers = {
            "original": img,
            "gray": cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR),
            "blur": cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR),
            "mask": cv2.cvtColor(obj_mask, cv2.COLOR_GRAY2BGR),
            "sobel": cv2.cvtColor(sobel_mag, cv2.COLOR_GRAY2BGR),
            "overlay": overlay
        }

        selected_view = view_layers.get(selected_layer, img).copy()
        contours, _, _ = detector_obj.detect_objects(img)

        record_data = []
        for i, cnt in enumerate(contours, start=1):
            if cm_per_pixel is None:
                continue
            results = compute_volume_and_error(cnt, cm_per_pixel)
            if not results:
                continue
            w_cm, h_cm, volume_cm3, err_total, (x, y, w, h) = results
            label = f"Object_{i}"
            color = (0, 255, 0)
            box = cv2.minAreaRect(cnt)
            box_points = np.intp(cv2.boxPoints(box))
            cv2.polylines(selected_view, [box_points], True, color, 2)
            cv2.putText(selected_view, f"{label}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(selected_view, f"W:{w_cm:.2f} H:{h_cm:.2f}", (x, y + h + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(selected_view, f"V:{volume_cm3:.1f} E:{err_total:.1f}%", (x, y + h + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            log_measurement(csv_filename, frame_idx, label, marker_status,
                            cm_per_pixel, w_cm, h_cm, volume_cm3, err_total)
            
            record_data.append({
                "id": label,
                "width": round(w_cm, 2),
                "height": round(h_cm, 2),
                "volume": round(volume_cm3, 1),
                "error": round(err_total, 1),
                "marker": marker_status
            })
        
        with lock:
            recent_records = record_data

        _, buffer = cv2.imencode('.jpg', selected_view)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html', title="Object Measurement using ArUco")


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/toggle', methods=['POST'])
def toggle():
    global running
    running = not running
    return jsonify({'running': running})


@app.route('/set_layer', methods=['POST'])
def set_layer():
    global selected_layer
    selected_layer = request.json.get('layer', 'original')
    return jsonify({'layer': selected_layer})


@app.route('/data')
def get_data():
    with lock:
        return jsonify(recent_records)


@app.route('/download')
def download_csv():
    return send_from_directory("Data", os.path.basename(csv_filename), as_attachment=True)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
