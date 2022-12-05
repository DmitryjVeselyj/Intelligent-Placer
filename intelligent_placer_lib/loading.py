import os
from cv2 import imread
import numpy as np


def load_images_from_folder(images_path: str) -> list[tuple[np.ndarray, str]]:
    images = []
    for filename in os.listdir(images_path):
        if filename.endswith(('.jpg', '.jpeg')):
            img = imread(os.path.join(images_path, filename))
            images.append((img, filename))
    return images


def load_image_from_path(image_path: str) -> tuple[np.ndarray, str]:
    head, filename = os.path.split(image_path)
    img = imread(image_path)
    return img, filename
