import cv2
from intelligent_placer_lib.preprocessing import preprocess_polygon, preprocess_things


def get_things_contours(image):
    img = preprocess_things(image)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def get_polygon_contour(image):
    img = preprocess_polygon(image)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return tuple()
    max_area_contour = max(contours, key=lambda x: cv2.contourArea(x))
    return max_area_contour
