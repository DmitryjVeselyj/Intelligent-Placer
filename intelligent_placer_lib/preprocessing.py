import cv2
import numpy as np
from skimage.feature import canny

def get_edges(image : np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)
    img = canny(img).astype(np.uint8)
    return img


def preprocess_polygon(image : np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)
    img = canny(img).astype(np.uint8)

    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    return img


def preprocess_things(image : np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)
    img = canny(img).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    img = cv2.dilate(img, kernel, iterations=3)
    img = cv2.erode(img, kernel, iterations=3)

    return img


 # находим нижнюю границу листа, чтобы в дальнейшем разбить изображение на две половины
def get_paper_line(image : np.ndarray) -> int:
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)
    img = canny(img).astype(np.uint8)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bboxs = map(cv2.boundingRect, contours) 
    max_area_bbox = max(bboxs, key=lambda x: x[2] * x[3]) # x y w h
    return max_area_bbox[1] + max_area_bbox[3] + 5