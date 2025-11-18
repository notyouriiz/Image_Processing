from flask import (
    Flask,
    render_template,
    Response,
    send_from_directory,
    jsonify,
    request,
)
import cv2, os, threading, numpy as np, io, time
from datetime import datetime
from measurement import (
    DetectorObj,
    init_csv_logger,
    log_measurement,
    compute_volume_and_error,
    preprocess_image,
)

app = Flask(__name__)

# Paths
os.makedirs("Data", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

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
selected_intensity = 1.0

csv_filename = init_csv_logger()
recent_records = []
lock = threading.Lock()

# image mode controls
mode = "camera"  # or "image"
image_list = []  # list of filepaths for image mode
image_index = 0
image_lock = threading.Lock()

# histogram bytes container (served as /histogram)
hist_lock = threading.Lock()

# keep latest processed view for histogram rendering
latest_processed_view = None
latest_processed_lock = threading.Lock()

# auto-tuning state
aruco_fail_count = 0
aruco_fail_threshold = 8
last_tune_time = 0
tune_cooldown = 3.0


# Pseudo color utilities
def safe_float(v, default=1.0):
    try:
        return float(v)
    except Exception:
        return default


_PSEUDO_MAP = {
    "pseudo_jet": cv2.COLORMAP_JET,
    "pseudo_hot": cv2.COLORMAP_HOT,
    "pseudo_turbo": cv2.COLORMAP_TURBO,
    "pseudo_magma": cv2.COLORMAP_MAGMA,
    "pseudo_inferno": cv2.COLORMAP_INFERNO,
    "pseudo_plasma": cv2.COLORMAP_PLASMA,
    "pseudo_cividis": cv2.COLORMAP_CIVIDIS,
    "pseudo_rainbow": cv2.COLORMAP_RAINBOW,
    "pseudo_bone": cv2.COLORMAP_BONE,
    "pseudo_ocean": cv2.COLORMAP_OCEAN,
}


def apply_pseudo_colormap(frame, mode_key, intensity=1.0):
    """
    Apply pseudo (thermal-like) colormap to the frame.
    intensity > 1.0 exaggerates highlights; intensity < 1.0 mutes.
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        f = np.clip(gray.astype(np.float32) / 255.0, 0.0, 1.0)
        if intensity <= 0:
            intensity = 1.0
        gamma = 1.0 / float(intensity)
        f = np.power(f, gamma)
        mean = np.clip(np.mean(f), 0.0001, 0.9999)
        f = (f - mean) * (1.0 + (intensity - 1.0) * 0.6) + mean
        f = np.clip(f, 0.0, 1.0)
        gray_enh = (f * 255.0).astype(np.uint8)
        cmap = _PSEUDO_MAP.get(mode_key, cv2.COLORMAP_TURBO)
        colored = cv2.applyColorMap(gray_enh, cmap)
        return colored
    except Exception:
        return frame


# existing helpers
def try_camera_auto_tune(frame):
    global cap, last_tune_time
    now = time.time()
    if now - last_tune_time < tune_cooldown:
        return frame
    last_tune_time = now
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    except Exception:
        pass
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
        sharpened = cv2.addWeighted(enhanced, 1.6, gaussian, -0.6, 0)
        return sharpened
    except Exception:
        return frame


def make_histogram_image_bytes(layer_img):
    """
    Create a histogram image (jpg bytes) for the provided image.
    If layer_img is color, produce RGB hist; if gray produce single-channel hist.
    """
    try:
        hist_h = 200
        hist_w = 320
        bins = 256
        canvas = np.zeros((hist_h, hist_w, 3), dtype=np.uint8) + 30

        if len(layer_img.shape) == 2:
            # grayscale
            hist = cv2.calcHist([layer_img], [0], None, [bins], [0, 256])
            cv2.normalize(hist, hist, 0, hist_h, cv2.NORM_MINMAX)
            hist = hist.flatten()
            for x in range(1, bins):
                x1 = int((x - 1) * hist_w / bins)
                x2 = int(x * hist_w / bins)
                y1 = int(hist_h - hist[x - 1])
                y2 = int(hist_h - hist[x])
                cv2.line(canvas, (x1, y1), (x2, y2), (200, 200, 200), 1)
        else:
            chans = cv2.split(layer_img)
            colors = [(200, 0, 0), (0, 200, 0), (0, 0, 200)]
            for i, ch in enumerate(chans):
                hist = cv2.calcHist([ch], [0], None, [bins], [0, 256])
                cv2.normalize(hist, hist, 0, hist_h, cv2.NORM_MINMAX)
                hist = hist.flatten()
                color = colors[i]
                for x in range(1, bins):
                    x1 = int((x - 1) * hist_w / bins)
                    x2 = int(x * hist_w / bins)
                    y1 = int(hist_h - hist[x - 1])
                    y2 = int(hist_h - hist[x])
                    cv2.line(canvas, (x1, y1), (x2, y2), color, 1)
        # add border
        border = cv2.copyMakeBorder(
            canvas, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(30, 30, 30)
        )
        _, buf = cv2.imencode(".jpg", border)
        return buf.tobytes()
    except Exception as e:
        print("histogram error:", e)
        placeholder = np.zeros((208, 328, 3), dtype=np.uint8) + 30
        _, p = cv2.imencode(".jpg", placeholder)
        return p.tobytes()


# Core processing (layer + pseudo support)
def process_frame_for_view(frame, layer_override=None, intensity_override=None):
    """
    Returns selected_view (BGR), marker_status, record_data (list)
    """
    global cm_per_pixel, aruco_fail_count, latest_processed_view, selected_layer, selected_intensity

    layer_use = layer_override if layer_override is not None else selected_layer
    intensity_use = safe_float(intensity_override, selected_intensity)

    img = frame.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # aruco detection
    corners, ids, _ = aruco_detector.detectMarkers(gray)
    marker_status = "No marker"
    detected_marker = False
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == marker_id_expected:
                cv2.polylines(img, [np.intp(corners[i][0])], True, (255, 0, 0), 2)
                marker_status = f"Marker OK (ID {marker_id})"
                side_px = np.median(
                    [
                        np.linalg.norm(corners[i][0][0] - corners[i][0][1]),
                        np.linalg.norm(corners[i][0][1] - corners[i][0][2]),
                        np.linalg.norm(corners[i][0][2] - corners[i][0][3]),
                        np.linalg.norm(corners[i][0][3] - corners[i][0][0]),
                    ]
                )
                if side_px > 10:
                    cm_per_pixel = marker_size_cm / side_px
                detected_marker = True
                break

    if not detected_marker:
        aruco_fail_count += 1
    else:
        aruco_fail_count = 0

    if aruco_fail_count >= aruco_fail_threshold:
        tuned = try_camera_auto_tune(frame)
        try:
            gray2 = cv2.cvtColor(tuned, cv2.COLOR_BGR2GRAY)
            corners2, ids2, _ = aruco_detector.detectMarkers(gray2)
            if ids2 is not None:
                for i, marker_id in enumerate(ids2.flatten()):
                    if marker_id == marker_id_expected:
                        cv2.polylines(
                            img, [np.intp(corners2[i][0])], True, (0, 128, 255), 2
                        )
                        marker_status = f"Marker OK (ID {marker_id}) [tuned]"
                        side_px = np.median(
                            [
                                np.linalg.norm(corners2[i][0][0] - corners2[i][0][1]),
                                np.linalg.norm(corners2[i][0][1] - corners2[i][0][2]),
                                np.linalg.norm(corners2[i][0][2] - corners2[i][0][3]),
                                np.linalg.norm(corners2[i][0][3] - corners2[i][0][0]),
                            ]
                        )
                        if side_px > 10:
                            cm_per_pixel = marker_size_cm / side_px
                        aruco_fail_count = 0
                        break
        except Exception:
            pass

    # Preprocessing
    overlay, gray_img, blurred, obj_mask, sobel_mag, dilated = preprocess_image(img)

    # build view layers
    view_layers = {
        "original": img,
        "gray": cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR),
        "blur": cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR),
        "mask": cv2.cvtColor(obj_mask, cv2.COLOR_GRAY2BGR),
        "sobel": cv2.cvtColor(sobel_mag, cv2.COLOR_GRAY2BGR),
        "overlay": overlay,
    }

    # if pseudo requested, compute that pseudo and add to view_layers
    if layer_use and layer_use.startswith("pseudo_"):
        pseudo_img = apply_pseudo_colormap(img, layer_use, intensity_use)
        view_layers[layer_use] = pseudo_img

    selected_view = view_layers.get(
        layer_use, view_layers.get(selected_layer, img)
    ).copy()

    # update the "latest_processed_view" used by histogram route
    with latest_processed_lock:
        try:
            latest_processed_view = selected_view.copy()
        except Exception:
            latest_processed_view = selected_view

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
        cv2.putText(
            selected_view,
            f"{label}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
        cv2.putText(
            selected_view,
            f"W:{w_cm:.2f} H:{h_cm:.2f}",
            (x, y + h + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
        cv2.putText(
            selected_view,
            f"V:{volume_cm3:.1f} E:{err_total:.1f}%",
            (x, y + h + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

        log_measurement(
            csv_filename,
            frame_idx,
            label,
            marker_status,
            cm_per_pixel,
            w_cm,
            h_cm,
            volume_cm3,
            err_total,
        )

        record_data.append(
            {
                "id": label,
                "width": round(w_cm, 2),
                "height": round(h_cm, 2),
                "volume": round(volume_cm3, 1),
                "error": round(err_total, 1),
                "marker": marker_status,
            }
        )

    return selected_view, marker_status, record_data


# Frame generator for camera (mjpeg)
def gen_frames():
    global frame_idx, cm_per_pixel, running, selected_layer, recent_records, mode, selected_intensity
    while True:
        try:
            layer_q = request.args.get("layer", None)
            intensity_q = request.args.get("intensity", None)
        except RuntimeError:
            layer_q = None
            intensity_q = None

        if mode != "camera":
            time.sleep(0.05)
            continue
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        if not running:
            _, buffer = cv2.imencode(".jpg", frame)
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )
            continue

        frame_idx += 1
        selected_view, marker_status, record_data = process_frame_for_view(
            frame, layer_override=layer_q, intensity_override=intensity_q
        )

        global recent_records
        with lock:
            recent_records = record_data

        _, buffer = cv2.imencode(".jpg", selected_view)
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html", title="Object Measurement using ArUco")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/image_feed")
def image_feed():
    global image_list, image_index, mode, frame_idx
    layer_q = request.args.get("layer", None)
    intensity_q = request.args.get("intensity", None)

    with image_lock:
        if not image_list:
            return ("No images uploaded or found in uploads/ folder."), 404
        path = image_list[image_index]
    frame = cv2.imread(path)
    if frame is None:
        return ("Image unreadable."), 500
    frame_idx += 1
    selected_view, marker_status, record_data = process_frame_for_view(
        frame, layer_override=layer_q, intensity_override=intensity_q
    )
    global recent_records
    with lock:
        recent_records = record_data
    _, buf = cv2.imencode(".jpg", selected_view)
    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.route("/histogram")
def histogram_jpg():
    """
    Serve histogram image (jpg) for the currently processed view.
    Query params:
      - layer
      - intensity
    """
    layer_q = request.args.get("layer", None)
    intensity_q = request.args.get("intensity", None)

    with latest_processed_lock:
        if latest_processed_view is None:
            placeholder = np.zeros((208, 328, 3), dtype=np.uint8) + 30
            _, p = cv2.imencode(".jpg", placeholder)
            return Response(p.tobytes(), mimetype="image/jpeg")
        # Use the stored latest_processed_view as base
        img = latest_processed_view.copy()

    if layer_q:
        if layer_q.startswith("pseudo_"):
            # intensity override if provided
            intensity_val = safe_float(intensity_q, selected_intensity)
            img = apply_pseudo_colormap(img, layer_q, intensity_val)
        else:
            # handle standard layers briefly for histogram view
            if layer_q == "gray":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif layer_q == "blur":
                gray_tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.GaussianBlur(gray_tmp, (9, 9), 0)
            elif layer_q == "sobel":
                gray_tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                sx = cv2.Sobel(gray_tmp, cv2.CV_64F, 1, 0, 3)
                sy = cv2.Sobel(gray_tmp, cv2.CV_64F, 0, 1, 3)
                mag = cv2.magnitude(sx, sy).astype(np.uint8)
                img = mag
            elif layer_q == "mask":
                gray_tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(
                    gray_tmp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                img = mask
            elif layer_q == "overlay":
                # approximate overlay by computing sobel and blending
                gray_tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                sx = cv2.Sobel(gray_tmp, cv2.CV_64F, 1, 0, 3)
                sy = cv2.Sobel(gray_tmp, cv2.CV_64F, 0, 1, 3)
                mag = cv2.magnitude(sx, sy).astype(np.uint8)
                overlay_img = cv2.addWeighted(
                    img, 0.8, cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR), 0.6, 0
                )
                img = overlay_img

    # now create histogram bytes
    hist_bytes = make_histogram_image_bytes(img)
    return Response(hist_bytes, mimetype="image/jpeg")


@app.route("/toggle", methods=["POST"])
def toggle():
    global running
    running = not running
    return jsonify({"running": running})


@app.route("/set_layer", methods=["POST"])
def set_layer():
    global selected_layer, selected_intensity
    payload = request.get_json() or {}
    selected_layer = payload.get("layer", selected_layer)
    if "intensity" in payload:
        selected_intensity = safe_float(payload.get("intensity", selected_intensity))
    return jsonify({"layer": selected_layer, "intensity": selected_intensity})


@app.route("/set_mode", methods=["POST"])
def set_mode():
    global mode
    requested = request.json.get("mode", "camera")
    if requested not in ("camera", "image", "folder"):
        return jsonify({"error": "invalid mode"}), 400
    mode = requested
    return jsonify({"mode": mode})


@app.route("/upload", methods=["POST"])
def upload():
    global image_list, image_index
    files = request.files.getlist("files")
    saved = []
    for f in files:
        if f and f.filename:
            safe_name = f.filename.replace(" ", "_")
            save_path = os.path.join("uploads", safe_name)
            f.save(save_path)
            saved.append(save_path)
    with image_lock:
        image_list.extend(saved)
        if image_list:
            image_index = len(image_list) - 1
    return jsonify({"saved": len(saved), "total_images": len(image_list)})


@app.route("/next_image")
def next_image():
    global image_index, image_list
    with image_lock:
        if not image_list:
            return jsonify({"error": "no images"}), 400
        image_index = (image_index + 1) % len(image_list)
        return jsonify({"index": image_index, "path": image_list[image_index]})


@app.route("/prev_image")
def prev_image():
    global image_index, image_list
    with image_lock:
        if not image_list:
            return jsonify({"error": "no images"}), 400
        image_index = (image_index - 1) % len(image_list)
        return jsonify({"index": image_index, "path": image_list[image_index]})


@app.route("/data")
def get_data():
    with lock:
        return jsonify(recent_records)


@app.route("/download")
def download_csv():
    return send_from_directory(
        "Data", os.path.basename(csv_filename), as_attachment=True
    )


if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(host="0.0.0.0", port=5000, debug=False)
