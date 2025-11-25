import cv2
import numpy as np


def apply_clahe(img, clip):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def apply_blur(img, val):
    k = int(val) * 2 + 1
    return cv2.GaussianBlur(img, (k, k), 0)


def apply_sharpen(img, strength):
    kernel = np.array([
        [0, -strength, 0],
        [-strength, 1 + strength * 4, -strength],
        [0, -strength, 0]
    ])
    return cv2.filter2D(img, -1, kernel)


def apply_denoise(img, strength):
    return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)


def compute_histogram(img):
    colors = ("b", "g", "r")
    hist_data = {}

    for i, c in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256]).flatten()
        hist_data[c] = hist.tolist()

    return hist_data
