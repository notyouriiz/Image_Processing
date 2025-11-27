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


def apply_segmentation_overlay(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Create a green overlay mask
    color_mask = np.zeros_like(img)
    color_mask[:, :, 1] = mask  # Green channel

    overlayed = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)
    return overlayed


def apply_alignment(img):
    h, w = img.shape[:2]

    # Example perspective transform
    src_pts = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    dst_pts = np.float32([[10, 10], [w-20, 5], [5, h-15], [w-10, h-10]])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    aligned = cv2.warpPerspective(img, matrix, (w, h))
    return aligned


def apply_frequency_mask(img, mask_radius=30):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)

    # Circular mask
    rows, cols = img_gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), mask_radius, 0, -1)

    fshift_masked = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_masked)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # Convert back to BGR
    return cv2.cvtColor(img_back.astype(np.uint8), cv2.COLOR_GRAY2BGR)