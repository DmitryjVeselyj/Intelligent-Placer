import pandas as pd
from intelligent_placer_lib import loading
from intelligent_placer_lib import intelligent_placer

if __name__ == '__main__':
    test_images_path = 'images/input/'
    test_images = loading.load_images_from_folder(test_images_path)
    df = pd.read_csv('images/input/ExpectedResultsWithDescr.csv',
                     delimiter=',', encoding="windows-1251")
    correct_cnt = 0
    for i, (image, im_name) in enumerate(test_images):
        print(im_name)
        expected_result = df.query(f'Name == \'{im_name}\'').iloc[0]['ExpectedResult']
        result = intelligent_placer.check_image(test_images_path + im_name)
        correct_cnt += (result == expected_result)
        print(
            f'{im_name}: Результат = {result}, Ожидаемый результат = {bool(expected_result)}')
    print(f'Правильных: {correct_cnt}\nВсего тестов: {len(test_images)}\nПроцент правильных ответов: {correct_cnt / len(test_images) * 100}%')