from intelligent_placer_lib.loading import load_image_from_path
from intelligent_placer_lib.recognizing import get_things_contours, get_polygon_contour
from intelligent_placer_lib.algorithm import the_best_algorithm_in_the_world


def check_image(image_path):
    min_height, max_height = 5, 600
    min_width, max_width = 0, 800

    image, image_name = load_image_from_path(image_path)
    polygon_image = image[min_height:max_height, min_width:max_width]
    things_image = image[max_height:image.shape[0] - 5, min_width:max_width]
    polygon_contour = get_polygon_contour(polygon_image)
    things_contours = get_things_contours(things_image)
    result = the_best_algorithm_in_the_world(polygon_contour, things_contours)
    return result
