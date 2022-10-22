import cv2
import numpy as np
from skimage.feature import canny


def preprocess_polygon(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)
    img = canny(img, sigma=1, low_threshold=0.1 * 255, high_threshold=0.2 * 255).astype(np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    return img


def preprocess_things(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)
    img = canny(img, sigma=3, low_threshold=10, high_threshold=30).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    img = cv2.dilate(img, kernel, iterations=3)
    img = cv2.erode(img, kernel, iterations=3)

    return img
