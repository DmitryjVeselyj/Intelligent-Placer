from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import rotate, translate
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean
from itertools import product
import scipy.optimize

N_ITER = 10 # число итераций основного алгоритма "just_an_algorithm".
MAX_DISTANCE_TO_BORDER = 30 # Используется для подсчёта числа точек, близких к границе
TOLERANCE = 1 # Погрешность площади (см "just_an_algorithm")


def distance_between_contours(polygon: Polygon | MultiPolygon, thing: Polygon) -> list[float]:
    if type(polygon) == Polygon:
        points_first = [*zip(*polygon.exterior.coords.xy)]
    else:
        points_first = []
        for poly in list(polygon.geoms):
            points_first += [*zip(*poly.exterior.coords.xy)]

    points_second = [*zip(*thing.exterior.coords.xy)]
    crosses = list(product(points_first, points_second))
    distances = [euclidean(points[0], points[1]) for points in crosses]
    return distances


def count_points(distances : list[float], value: int) -> int:
    cnt = list(filter(lambda x: x < value, distances))
    return len(cnt)


def get_begin_position(polygon : Polygon, thing : Polygon) -> tuple[float, float]:
    thing_xc, thing_yc = thing.centroid.coords.xy
    poly_xc, poly_yc = polygon.centroid.coords.xy
    return poly_xc[0] - thing_xc[0], poly_yc[0] - thing_yc[0]


def get_optimal_position(polygon : Polygon, thing: Polygon) -> tuple[float, float, float, float]:
    def f(args):
        rot, xoff, yoff = args
        transformed_thing = rotate(translate(thing, xoff=xoff, yoff=yoff), rot)
        distances = distance_between_contours(polygon, transformed_thing)
        thing_xc, thing_yc = transformed_thing.centroid.coords.xy
        poly_xc, poly_yc = polygon.centroid.coords.xy
        """
        1. Число точек, находящихся на расстоянии меньше MAX_DISTANCE. Чем больше, тем плотнее к границе
        2. Пересечение многоугольника и предмета. Сделано, чтобы в результате преобразований оставались внутри многоугольника или хотя бы не отдалялись от него
        3. Расстояние между центрами многоугольника и предмета. Так я пытался дополнительно усилить уплотнение к границе и заполнять пропуски в многоугольнике
        """
        return -count_points(distances, MAX_DISTANCE_TO_BORDER) - polygon.intersection(transformed_thing).area - euclidean((poly_xc[0], poly_yc[0]), (thing_xc[0], thing_yc[0]))

    minx, miny, maxx, maxy = map(int, polygon.bounds)
    w = maxx - minx
    h = maxy - miny
    result = scipy.optimize.differential_evolution(f, bounds=((0, 360), (-w, w), (-h, h)))

    rot, xoff, yoff = result.x
    fopt = result.fun

    return rot, xoff, yoff, fopt


def just_an_algorithm(polygon: Polygon, things: tuple[Polygon]) -> bool:
    if polygon.is_empty or len(things) == 0:
        return False
    # if polygon.area < sum([obj.area for obj in things]):
    #     return False
    params = []
    tmp_polygon = Polygon(polygon)
    x, y = tmp_polygon.exterior.coords.xy #можно убрать. Добавлено, чтобы в юпитерском файле построились графики
    plt.plot(x, y) # можно убрать
    flag = True  # Соответственно, если не нужно рисовать графики, можно сделать и без флага. Тогда алгоритм будет быстрее выдавать результат
    for thing in things:
        xoff0, yoff0 = get_begin_position(tmp_polygon, thing)
        tmp_thing = translate(thing, xoff=xoff0, yoff=yoff0)
        for _ in range(N_ITER):
            rot, xoff, yoff, fopt = get_optimal_position(tmp_polygon, tmp_thing)
            params.append([rot, xoff, yoff, fopt])
        rot, xoff, yoff, fopt = min(params, key=lambda x: x[3])
        tmp_thing = rotate(translate(tmp_thing, xoff=xoff, yoff=yoff), rot)
        x, y = tmp_thing.exterior.coords.xy  # можно убрать
        plt.plot(x, y)  # можно убрать
        if tmp_thing.difference(tmp_polygon).area > TOLERANCE:
            flag = False

        tmp_polygon = tmp_polygon.difference(tmp_thing)
        params.clear()
    plt.show()
    return flag
