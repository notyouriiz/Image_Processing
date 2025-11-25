from flask import Flask, render_template, request, jsonify
from processing import (
    apply_clahe, apply_blur, apply_sharpen,
    apply_denoise, compute_histogram
)
import base64
import cv2
import numpy as np
import os

app = Flask(__name__)
uploaded_images = {}
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def to_base64(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


@app.route("/", methods=["GET", "POST"])
def index():
    global uploaded_images

    if request.method == "POST":
        files = request.files.getlist("images")
        uploaded_images = {}

        for f in files:
            img_array = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            uploaded_images[f.filename] = img

            # Save original image for preview
            cv2.imwrite(os.path.join(UPLOAD_FOLDER, f.filename), img)

    return render_template("index.html", images=list(uploaded_images.keys()))


@app.route("/process", methods=["POST"])
def process():
    filename = request.form.get("filename")
    clahe_val = float(request.form.get("clahe"))
    blur_val = float(request.form.get("blur"))
    sharp_val = float(request.form.get("sharp"))
    denoise_val = float(request.form.get("denoise"))

    img = uploaded_images[filename].copy()

    # Apply enhancement
    if clahe_val > 0:
        img = apply_clahe(img, clahe_val)

    if blur_val > 0:
        img = apply_blur(img, blur_val)

    if sharp_val > 0:
        img = apply_sharpen(img, sharp_val)

    if denoise_val > 0:
        img = apply_denoise(img, denoise_val)

    # Histogram
    hist = compute_histogram(img)

    return jsonify({
        "image": to_base64(img),
        "hist": hist
    })

@app.route("/histogram")
def histogram():
    image_path = request.args.get("image")
    if image_path is None:
        return {"error": "No image provided"}, 400

    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Cannot read image"}, 400

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten().tolist()

    return {"hist": hist}


if __name__ == "__main__":
    app.run(debug=True)
