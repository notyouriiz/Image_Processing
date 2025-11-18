from flask import Flask, render_template, Response, send_from_directory, jsonify, request, send_file
import cv2, os, threading, numpy as np, io, time
from datetime import datetime
from measurement import DetectorObj, init_csv_logger, log_measurement, compute_volume_and_error, preprocess_image

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
selected_intensity = 1.0  # default pseudo intensity

csv_filename = init_csv_logger()
recent_records = []
lock = threading.Lock()

# image mode controls
mode = "camera"  # or "image"
image_list = []  # list of filepaths for image mode
image_index = 0
image_lock = threading.Lock()

# histogram bytes container (served as /histogram.jpg)
hist_bytes = None
hist_lock = threading.Lock()

# auto-tuning state for Type C (ArUco reliability)
aruco_fail_count = 0
aruco_fail_threshold = 8  # after these many consecutive fails, try auto-tune
last_tune_time = 0
tune_cooldown = 3.0  # seconds

# -------------------------
# Pseudo color utilities
# -------------------------
def safe_float(v, default=1.0):
    try:
        return float(v)
    except Exception:
        return default

# Map friendly layer names to OpenCV colormap constants
_PSEUDO_MAP = {
    "pseudo_jet": cv2.COLORMAP_JET,
    "pseudo_hot": cv2.COLORMAP_HOT,
    "pseudo_turbo": cv2.COLORMAP_TURBO,
    "pseudo_magma": cv2.COLORMAP_MAGMA,
    "pseudo_inferno": cv2.COLORMAP_INFERNO,
    "pseudo_plasma": cv2.COLORMAP_PLASMA,
    "pseudo_cividis": cv2.COLORMAP_CIVIDIS,
}

def apply_pseudo_colormap(frame, mode_key, intensity=1.0):
    """
    Apply pseudo (thermal-like) colormap to the frame.
    intensity > 1.0 exaggerates highlights; intensity < 1.0 mutes.
    mode_key selects which OpenCV colormap to use (keys from _PSEUDO_MAP).
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # intensity manipulation:
        # scale to [0,1], apply gamma-like transform controlled by intensity,
        # then rescale to [0,255]
        # Using exponent 1/intensity so slider >1 brightens high values.
        f = np.clip(gray.astype(np.float32) / 255.0, 0.0, 1.0)
        if intensity <= 0:
            intensity = 1.0
        # Use a soft transform combining power and contrast
        gamma = 1.0 / float(intensity)
        f = np.power(f, gamma)
        # optional slight contrast stretch based on intensity
        mean = np.clip(np.mean(f), 0.0001, 0.9999)
        f = (f - mean) * (1.0 + (intensity - 1.0) * 0.6) + mean
        f = np.clip(f, 0.0, 1.0)
        gray_enh = (f * 255.0).astype(np.uint8)

        cmap = _PSEUDO_MAP.get(mode_key, cv2.COLORMAP_TURBO)
        colored = cv2.applyColorMap(gray_enh, cmap)
        return colored
    except Exception:
        # on any failure, return a safe fallback (original frame)
        return frame

# -------------------------
# Existing helpers (auto-tune, histogram, etc.)
# -------------------------
def try_camera_auto_tune(frame):
    """
    Try some camera property adjustments and image-based enhancements to improve ArUco detection.
    This is Type C auto-tuning: target ArUco reliability.
    Returns a modified frame (may be sharpened / CLAHE applied) to use for detection attempts.
    """
    global cap, last_tune_time
    now = time.time()
    # Don't tune too frequently
    if now - last_tune_time < tune_cooldown:
        return frame
    last_tune_time = now

    # Try toggling some camera properties if supported
    try:
        # Many backends use CAP_PROP_AUTO_EXPOSURE: 0.25 or 1.0 values differ by OS.
        # Try toggling off/on quickly to let camera reauto-adjust.
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # attempt
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    except Exception:
        pass

    # Try small sharpening + CLAHE to improve marker edges
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        # Unsharp mask
        gaussian = cv2.GaussianBlur(enhanced, (0,0), sigmaX=1.0)
        sharpened = cv2.addWeighted(enhanced, 1.6, gaussian, -0.6, 0)

        return sharpened
    except Exception:
        return frame

def make_histogram_image_for_layer(layer_img, layer_name=None, intensity=1.0):
    """
    Given a BGR image (or single-channel), create a histogram image (as JPEG bytes).
    If layer_name indicates a pseudo map, compute histogram from the grayscale base.
    """
    global hist_bytes, hist_lock
    try:
        # If it's a pseudo layer, compute histogram on its grayscale base
        if layer_name and layer_name.startswith("pseudo_"):
            # Convert back to gray for histogram generation consistently
            gray_for_hist = cv2.cvtColor(layer_img, cv2.COLOR_BGR2GRAY)
            chans = [gray_for_hist]
            colors = [(200,200,200)]
        else:
            if len(layer_img.shape) == 2:
                gray = layer_img
                chans = [gray]
                colors = [(200,200,200)]
            else:
                b,g,r = cv2.split(layer_img)
                chans = [b,g,r]
                colors = [(200,0,0),(0,200,0),(0,0,200)]
        hist_h = 200
        hist_w = 320
        bin_count = 256
        hist_img = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

        for i, ch in enumerate(chans):
            hist = cv2.calcHist([ch], [0], None, [bin_count], [0,256])
            cv2.normalize(hist, hist, 0, hist_h, cv2.NORM_MINMAX)
            hist = hist.flatten()
            color = colors[i]
            for x in range(1, bin_count):
                x1 = int((x-1) * hist_w / bin_count)
                x2 = int(x * hist_w / bin_count)
                y1 = int(hist_h - hist[x-1])
                y2 = int(hist_h - hist[x])
                cv2.line(hist_img, (x1,y1), (x2,y2), color, 1)
        # Add border / background subtle
        border = cv2.copyMakeBorder(hist_img, 4,4,4,4, cv2.BORDER_CONSTANT, value=(30,30,30))
        _, buf = cv2.imencode('.jpg', border)
        with hist_lock:
            hist_bytes = buf.tobytes()
        return True
    except Exception as e:
        print("make_histogram error:", e)
        return False

# -------------------------
# Core processing (updated to accept layer/intensity)
# -------------------------
def process_frame_for_view(frame, layer_override=None, intensity_override=None):
    """
    Run ArUco detection, preprocessing and object detection on a single BGR frame.
    Returns selected_view (BGR), marker_status, record_data (list), and optionally updates histogram.
    layer_override and intensity_override are optional; if not provided fall back to global selected_layer / intensity.
    """
    global cm_per_pixel, aruco_fail_count, hist_bytes, selected_layer, selected_intensity

    # determine layer and intensity to use for rendering/histogram
    layer_use = layer_override if layer_override is not None else selected_layer
    intensity_use = safe_float(intensity_override, selected_intensity)

    img = frame.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # primary aruco detection; if fails repeatedly, invoke auto-tuning then reattempt
    corners, ids, _ = aruco_detector.detectMarkers(gray)
    marker_status = "No marker"
    detected_marker = False
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
                detected_marker = True
                break

    if not detected_marker:
        aruco_fail_count += 1
    else:
        aruco_fail_count = 0

    # If too many fails, try auto-tune and re-detect on tuned frame
    if aruco_fail_count >= aruco_fail_threshold:
        tuned = try_camera_auto_tune(frame)
        try:
            gray2 = cv2.cvtColor(tuned, cv2.COLOR_BGR2GRAY)
            corners2, ids2, _ = aruco_detector.detectMarkers(gray2)
            if ids2 is not None:
                for i, marker_id in enumerate(ids2.flatten()):
                    if marker_id == marker_id_expected:
                        cv2.polylines(img, [np.intp(corners2[i][0])], True, (0, 128, 255), 2)
                        marker_status = f"Marker OK (ID {marker_id}) [tuned]"
                        side_px = np.median([
                            np.linalg.norm(corners2[i][0][0] - corners2[i][0][1]),
                            np.linalg.norm(corners2[i][0][1] - corners2[i][0][2]),
                            np.linalg.norm(corners2[i][0][2] - corners2[i][0][3]),
                            np.linalg.norm(corners2[i][0][3] - corners2[i][0][0]),
                        ])
                        if side_px > 10:
                            cm_per_pixel = marker_size_cm / side_px
                        aruco_fail_count = 0
                        break
        except Exception:
            pass

    # Preprocessing
    overlay, gray_img, blurred, obj_mask, sobel_mag, dilated = preprocess_image(img)

    # prepare pseudo layers if requested
    pseudo_layers = {}
    for key in _PSEUDO_MAP.keys():
        # we compute pseudo maps lazily only if requested by layer_use
        if layer_use == key:
            pseudo_layers[key] = apply_pseudo_colormap(img, key, intensity_use)

    # Build view layers (including pseudo if requested)
    view_layers = {
        "original": img,
        "gray": cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR),
        "blur": cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR),
        "mask": cv2.cvtColor(obj_mask, cv2.COLOR_GRAY2BGR),
        "sobel": cv2.cvtColor(sobel_mag, cv2.COLOR_GRAY2BGR),
        "overlay": overlay
    }
    # merge pseudo if the selected layer is a pseudo key
    if layer_use in pseudo_layers:
        view_layers[layer_use] = pseudo_layers[layer_use]

    selected_view = view_layers.get(layer_use, view_layers.get(selected_layer, img)).copy()

    # compute histogram for the selected layer (background thread could also be used)
    try:
        # If pseudo used, pass its name so histogram can compute on grayscale base
        make_histogram_image_for_layer(view_layers.get(layer_use, img), layer_name=layer_use, intensity=intensity_use)
    except Exception:
        pass

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

    return selected_view, marker_status, record_data

# Frame generator for camera (mjpeg)
def gen_frames():
    global frame_idx, cm_per_pixel, running, selected_layer, recent_records, mode, selected_intensity
    while True:
        # For each client call we respect optional query parameters `layer` and `intensity`
        # These are read from the request context (works while the generator runs under Flask)
        try:
            layer_q = request.args.get('layer', None)
            intensity_q = request.args.get('intensity', None)
        except RuntimeError:
            # no request context - fallback to globals
            layer_q = None
            intensity_q = None

        if mode != "camera":
            time.sleep(0.05)
            continue
        ret, frame = cap.read()
        if not ret:
            # camera failure: yield nothing for a short while
            time.sleep(0.1)
            continue

        if not running:
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            continue

        frame_idx += 1
        selected_view, marker_status, record_data = process_frame_for_view(frame, layer_override=layer_q, intensity_override=intensity_q)

        global recent_records
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

@app.route('/image_feed')
def image_feed():
    """
    Serve the current image (processed) for image mode as a single jpeg. The frontend should poll this.
    Supports optional query params:
      - layer: name of layer to render (e.g. original, gray, pseudo_turbo)
      - intensity: float pseudo intensity (e.g. 1.0)
    """
    global image_list, image_index, mode, frame_idx
    layer_q = request.args.get('layer', None)
    intensity_q = request.args.get('intensity', None)

    with image_lock:
        if not image_list:
            return ("No images uploaded or found in uploads/ folder."), 404
        # load current image
        path = image_list[image_index]
    frame = cv2.imread(path)
    if frame is None:
        return ("Image unreadable."), 500
    frame_idx += 1
    selected_view, marker_status, record_data = process_frame_for_view(frame, layer_override=layer_q, intensity_override=intensity_q)
    global recent_records
    with lock:
        recent_records = record_data
    _, buf = cv2.imencode('.jpg', selected_view)
    return Response(buf.tobytes(), mimetype='image/jpeg')

@app.route('/histogram.jpg')
def histogram_jpg():
    """
    Serve last generated histogram JPEG bytes.
    Query params:
      - layer (optional)
      - intensity (optional)
    """
    global hist_bytes
    layer_q = request.args.get('layer', None)
    intensity_q = request.args.get('intensity', None)

    # force one histogram generation pass if needed by calling process_frame_for_view on a small buffer
    # but prefer to serve latest cached histogram
    with hist_lock:
        if hist_bytes is None:
            placeholder = np.zeros((208,328,3), dtype=np.uint8) + 30
            _, p = cv2.imencode('.jpg', placeholder)
            return Response(p.tobytes(), mimetype='image/jpeg')
        return Response(hist_bytes, mimetype='image/jpeg')

@app.route('/toggle', methods=['POST'])
def toggle():
    global running
    running = not running
    return jsonify({'running': running})

@app.route('/set_layer', methods=['POST'])
def set_layer():
    global selected_layer, selected_intensity
    # support clients sending intensity as well
    payload = request.get_json() or {}
    selected_layer = payload.get('layer', selected_layer)
    # Accept intensity as float if provided; otherwise keep previous
    if 'intensity' in payload:
        selected_intensity = safe_float(payload.get('intensity', selected_intensity), selected_intensity)
    return jsonify({'layer': selected_layer, 'intensity': selected_intensity})

@app.route('/set_mode', methods=['POST'])
def set_mode():
    """
    set mode to 'camera' or 'image' or 'folder'
    """
    global mode
    requested = request.json.get('mode', 'camera')
    if requested not in ('camera', 'image', 'folder'):
        return jsonify({'error': 'invalid mode'}), 400
    mode = requested
    return jsonify({'mode': mode})

@app.route('/upload', methods=['POST'])
def upload():
    """
    Accept multiple files via form-data (input name 'files'), save into 'uploads' and append to image_list.
    """
    global image_list, image_index
    files = request.files.getlist('files')
    saved = []
    for f in files:
        if f and f.filename:
            safe_name = f.filename.replace(" ", "_")
            save_path = os.path.join("uploads", safe_name)
            f.save(save_path)
            saved.append(save_path)
    with image_lock:
        # append saved files to image_list and set index to last uploaded
        image_list.extend(saved)
        if image_list:
            image_index = len(image_list) - 1
    return jsonify({'saved': len(saved), 'total_images': len(image_list)})


@app.route('/next_image')
def next_image():
    global image_index, image_list
    with image_lock:
        if not image_list:
            return jsonify({'error': 'no images'}), 400
        image_index = (image_index + 1) % len(image_list)
        return jsonify({'index': image_index, 'path': image_list[image_index]})

@app.route('/prev_image')
def prev_image():
    global image_index, image_list
    with image_lock:
        if not image_list:
            return jsonify({'error': 'no images'}), 400
        image_index = (image_index - 1) % len(image_list)
        return jsonify({'index': image_index, 'path': image_list[image_index]})

@app.route('/data')
def get_data():
    with lock:
        return jsonify(recent_records)

@app.route('/download')
def download_csv():
    return send_from_directory("Data", os.path.basename(csv_filename), as_attachment=True)

if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(host='0.0.0.0', port=5000, debug=False)
