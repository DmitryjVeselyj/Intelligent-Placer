import os
from cv2 import imread

def load_images_from_folder(images_path):
    images = []
    for filename in os.listdir(images_path):
        if filename.endswith(('.jpg', '.jpeg')):
            img = imread(os.path.join(images_path, filename))
            images.append((img, filename))
    return images


def load_image_from_path(image_path):
    head, filename = os.path.split(image_path)
    img = imread(image_path)
    return img, filename


def get_images(images_path):
    if images_path.endswith(('.jpg', '.jpeg')):
        test_images = [load_image_from_path(images_path)]
    else:
        test_images = load_images_from_folder(images_path)
    return test_images
