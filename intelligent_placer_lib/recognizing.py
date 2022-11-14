import cv2
from intelligent_placer_lib.preprocessing import preprocess_polygon, preprocess_things
from shapely.geometry import Polygon
import numpy as np

MIN_AREA = 50 # для фильтра шумов
SIMPLIFY_TOLERANCE = 3 # Тк у контура слишком много точек, мы уменьшаем их количество. Можно и 0 поставить, но так ждать дольше


def get_max_size(contour: np.ndarray) -> int:
    x, y, w, h = cv2.boundingRect(contour)
    return max(w, h)

def get_things_contours(image : np.ndarray) -> tuple[Polygon]:
    img = preprocess_things(image)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = filter(lambda x: cv2.contourArea(x) > MIN_AREA, contours)     
    contours = sorted(contours, key=lambda x: get_max_size(x), reverse=True) 
    contours = tuple(Polygon(cnt[:, 0]).simplify(SIMPLIFY_TOLERANCE) for cnt in contours)
    return contours


def get_polygon_contour(image: np.ndarray) -> Polygon:
    img = preprocess_polygon(image)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = tuple(filter(lambda x: cv2.contourArea(x) > MIN_AREA, contours))
    if len(contours) == 0:
        return Polygon()
    max_area_contour = max(contours, key=lambda x: cv2.contourArea(x))
    max_area_contour = Polygon(max_area_contour[:, 0]).simplify(SIMPLIFY_TOLERANCE)
    return max_area_contour



