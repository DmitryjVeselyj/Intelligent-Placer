import cv2
from random import choice


def the_best_algorithm_in_the_world(polygon, things):
    if len(polygon) == 0 or len(things) == 0:
        return False
    if cv2.contourArea(polygon) < sum([cv2.contourArea(obj) for obj in things]):
        return False
    return choice([False, True])
