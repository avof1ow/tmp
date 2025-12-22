import cProfile
import pstats
import cv2
import numpy as np
from pysift import computeKeypointsAndDescriptors


def profile_sift():
    # Создаем тестовое изображение
    image = np.random.rand(512, 512) * 255
    image = image.astype(np.uint8)

    # Профилируем основную функцию
    cProfile.runctx('computeKeypointsAndDescriptors(image)',
                    globals(), locals(), 'sift_profile.prof')

    # Анализируем результаты
    stats = pstats.Stats('sift_profile.prof')
    stats.strip_dirs().sort_stats('time').print_stats(20)
    stats.strip_dirs().sort_stats('cumulative').print_stats(20)

    # Сохраняем в файл
    stats.dump_stats('sift_profile.prof')


if __name__ == '__main__':
    profile_sift()