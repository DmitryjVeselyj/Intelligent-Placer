from intelligent_placer_lib.loading import load_image_from_path
from intelligent_placer_lib.recognizing import get_things_contours, get_polygon_contour
from intelligent_placer_lib.preprocessing import get_paper_line
from intelligent_placer_lib.algorithm import just_an_algorithm


def check_image(image_path : str) -> bool:
    min_height = 5
    min_width= 0 

    image, image_name = load_image_from_path(image_path)

    dividing_line = get_paper_line(image)
    polygon_image = image[min_height:dividing_line, min_width:image.shape[1] - 5]
    things_image = image[dividing_line:image.shape[0] - 5, min_width:image.shape[1]- 5]
    
    polygon_contour = get_polygon_contour(polygon_image)
    things_contours = get_things_contours(things_image)
    result = just_an_algorithm(polygon_contour, things_contours)
    return result
