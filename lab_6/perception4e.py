"""
Unit tests for perception4e.py
"""
import pytest
import numpy as np
import sys
import os

# Добавляем путь к исходному коду
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from perception4e import (
    array_normalization,
    gradient_edge_detector,
    gaussian_derivative_edge_detector,
    laplacian_edge_detector,
    gen_gray_scale_picture,
    sum_squared_difference,
    probability_contour_detection,
    gen_discs
)


class TestArrayNormalization:
    """Тесты для функции нормализации массивов"""

    def test_normalization_basic(self):
        """Тест базовой нормализации"""
        # Arrange
        data = np.array([1, 2, 3, 4, 5])

        # Act
        result = array_normalization(data, 0, 1)

        # Assert
        assert result.min() == 0
        assert result.max() == 1
        assert np.allclose(result, [0.0, 0.25, 0.5, 0.75, 1.0])

    def test_normalization_negative_values(self):
        """Тест нормализации с отрицательными значениями"""
        # Arrange
        data = np.array([-5, 0, 5])

        # Act
        result = array_normalization(data, 0, 100)

        # Assert
        assert result.min() == 0
        assert result.max() == 100
        assert np.allclose(result, [0.0, 50.0, 100.0])

    def test_normalization_single_value(self):
        """Тест нормализации с одним уникальным значением"""
        # Arrange
        data = np.array([5, 5, 5])

        # Act
        result = array_normalization(data, 10, 20)

        # Assert
        assert result[0] == 10  # Все значения должны быть равны range_min
        assert np.all(result == 10)

    def test_normalization_with_list(self):
        """Тест что функция работает с list на входе"""
        # Arrange
        data = [1, 2, 3, 4]

        # Act
        result = array_normalization(data, 0, 255)

        # Assert
        assert isinstance(result, np.ndarray)
        assert result.min() == 0
        assert result.max() == 255


class TestGrayScalePictureGeneration:
    """Тесты для генерации изображений в оттенках серого"""

    def test_gen_gray_scale_picture_basic(self):
        """Тест базовой генерации изображения"""
        # Arrange
        size = 5
        level = 3

        # Act
        result = gen_gray_scale_picture(size, level)

        # Assert
        assert result.shape == (size, size)
        assert result.dtype == np.float64
        assert result.min() >= 0
        assert result.max() <= 255

    def test_gen_gray_scale_picture_single_level(self):
        """Тест генерации с одним уровнем серого"""
        # Arrange
        size = 3

        # Act
        result = gen_gray_scale_picture(size, level=1)

        # Assert
        assert result.shape == (size, size)
        assert np.all(result == 0)  # Все пиксели должны быть черными

    def test_gen_gray_scale_picture_invalid_level(self):
        """Тест с некорректным уровнем"""
        # Arrange
        size = 5

        # Act & Assert
        with pytest.raises(AssertionError):
            gen_gray_scale_picture(size, level=0)

    def test_gen_gray_scale_picture_varying_intensity(self):
        """Тест что интенсивность изменяется по изображению"""
        # Arrange
        size = 4
        level = 2

        # Act
        result = gen_gray_scale_picture(size, level)

        # Assert
        # Проверяем что значения различаются
        unique_values = np.unique(result)
        assert len(unique_values) > 1


class TestDiscGeneration:
    """Тесты для генерации дисков"""

    def test_gen_discs_basic(self):
        """Тест базовой генерации дисков"""
        # Arrange
        init_scale = 3
        scales = 2

        # Act
        discs = gen_discs(init_scale, scales)

        # Assert
        assert len(discs) == scales
        # Для каждого масштаба должно быть 8 дисков
        assert len(discs[0]) == 8
        assert len(discs[1]) == 8

        # Проверяем форму дисков
        for scale_discs in discs:
            for disc in scale_discs:
                assert disc.shape == (init_scale, init_scale)
                # Диски должны содержать только 0 и 255
                assert np.all(np.isin(disc, [0, 255]))

    def test_gen_discs_single_scale(self):
        """Тест генерации одного масштаба"""
        # Arrange
        init_scale = 5

        # Act
        discs = gen_discs(init_scale, scales=1)

        # Assert
        assert len(discs) == 1
        assert len(discs[0]) == 8

        # Проверяем что диски различаются
        disc_values = [disc.tobytes() for disc in discs[0]]
        assert len(set(disc_values)) == 8  # Все диски должны быть уникальными

    def test_gen_discs_symmetry(self):
        """Тест симметрии дисков"""
        # Arrange
        init_scale = 3

        # Act
        discs = gen_discs(init_scale, scales=1)
        first_scale_discs = discs[0]

        # Assert
        # Проверяем пары дисков (верхний/нижний, левый/правый и т.д.)
        lower_half = first_scale_discs[0]
        upper_half = first_scale_discs[1]

        # Верхняя половина должна быть отражением нижней
        assert np.array_equal(upper_half, np.flip(lower_half, axis=0))

        left_half = first_scale_discs[2]
        right_half = first_scale_discs[3]

        # Правая половина должна быть отражением левой
        assert np.array_equal(right_half, np.flip(left_half, axis=1))


class TestSSDFunction:
    """Тесты для функции суммы квадратов разностей"""

    def test_ssd_identical_images(self):
        """Тест SSD для идентичных изображений"""
        # Arrange
        img1 = np.random.rand(10, 10)
        img2 = img1.copy()  # Точная копия

        # Act
        shift, ssd = sum_squared_difference(img1, img2)

        # Assert
        assert shift == (0, 0)  # Нет сдвига для идентичных изображений
        assert ssd == 0  # Разность должна быть 0

    def test_ssd_shifted_images(self):
        """Тест SSD для сдвинутых изображений"""
        # Arrange
        img1 = np.random.rand(20, 20)
        # Сдвигаем на (5, 3)
        img2 = np.roll(img1, 5, axis=0)
        img2 = np.roll(img2, 3, axis=1)

        # Act
        shift, ssd = sum_squared_difference(img1, img2)

        # Assert
        # Должен найти сдвиг (5, 3) или близкий к нему
        assert shift == (5, 3)
        assert ssd == 0  # Так как это точный сдвиг

    def test_ssd_different_shapes_error(self):
        """Тест что функция вызывает ошибку при разных размерах"""
        # Arrange
        img1 = np.random.rand(10, 10)
        img2 = np.random.rand(8, 8)

        # Act & Assert
        with pytest.raises(AssertionError):
            sum_squared_difference(img1, img2)

    def test_ssd_with_noise(self):
        """Тест SSD с зашумленным изображением"""
        # Arrange
        img1 = np.random.rand(15, 15)
        img2 = img1 + np.random.normal(0, 0.1, img1.shape)  # Добавляем шум

        # Act
        shift, ssd = sum_squared_difference(img1, img2)

        # Assert
        # Сдвиг должен быть (0, 0) так как мы не сдвигали
        assert shift == (0, 0)
        # SSD должен быть маленьким но не нулевым
        assert ssd > 0
        assert ssd < 100  # Эвристическая проверка


class TestProbabilityContourDetection:
    """Тесты для обнаружения контуров"""

    def test_probability_contour_empty_image(self):
        """Тест обнаружения контуров на пустом изображении"""
        # Arrange
        image = np.zeros((10, 10))
        discs = gen_discs(3, 1)[0]  # Берем диски первого масштаба

        # Act
        result = probability_contour_detection(image, discs, threshold=0)

        # Assert
        assert result.shape == image.shape
        assert np.all(result == 0)  # На пустом изображении не должно быть контуров

    def test_probability_contour_edge_image(self):
        """Тест обнаружения контуров на изображении с краем"""
        # Arrange
        image = np.zeros((10, 10))
        image[:, 5:] = 255  # Резкий переход посередине
        discs = gen_discs(3, 1)[0]

        # Act
        result = probability_contour_detection(image, discs, threshold=0)

        # Assert
        assert result.shape == image.shape
        # Должны быть обнаружены контуры в области перехода
        assert np.any(result > 0)

        # Контуры должны быть в районе x=4-6 (центр диска)
        edge_region = result[4:7, :]
        assert np.sum(edge_region > 0) > 0

    def test_probability_contour_threshold(self):
        """Тест влияния порога на обнаружение контуров"""
        # Arrange
        image = np.random.rand(8, 8) * 100  # Случайное изображение
        discs = gen_discs(3, 1)[0]

        # Act
        result_low_thresh = probability_contour_detection(image, discs, threshold=0)
        result_high_thresh = probability_contour_detection(image, discs, threshold=1000)

        # Assert
        # С высоким порогом должно быть меньше контуров
        low_count = np.sum(result_low_thresh > 0)
        high_count = np.sum(result_high_thresh > 0)
        assert low_count >= high_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])