import cv2
from intelligent_placer_lib.preprocessing import preprocess_polygon, preprocess_things
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
from shapely.affinity import translate

MIN_AREA = 50

def get_things_contours(image):
    img = preprocess_things(image)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = filter(lambda x: cv2.contourArea(x) > MIN_AREA, contours)
    contours = tuple(Polygon(cnt[:, 0]) for cnt in contours)
    return contours


def get_polygon_contour(image):
    img = preprocess_polygon(image)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = tuple(filter(lambda x: cv2.contourArea(x) > MIN_AREA, contours))
    if len(contours) == 0:
        return Polygon()
    max_area_contour = max(contours, key=lambda x: cv2.contourArea(x))
    max_area_contour = Polygon(max_area_contour[:, 0])
    return max_area_contour

def center_contour(contour, point=(0, 0)):
    c_x, c_y = contour.centroid.coords.xy
    cnt = translate(contour, xoff=point[0] - c_x[0], yoff=point[1] - c_y[0])
    return cnt

