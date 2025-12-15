"""Тесты для мок-версии perception4e.py"""
import pytest
import numpy as np
import sys
import os
from unittest.mock import patch

# Используем мок-версию
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Импортируем из мок-версии
from perception4e_mock import (
    array_normalization,
    gradient_edge_detector,
    gaussian_derivative_edge_detector,
    laplacian_edge_detector,
    gen_gray_scale_picture,
    sum_squared_difference,
    probability_contour_detection,
    gen_discs,
    pool_rois,
    pool_roi
)


# ============================================================================
# БАЗОВЫЕ ТЕСТЫ (минимум 7 тестов)
# ============================================================================

def test_1_array_normalization():
    """Тест 1: Нормализация массива"""
    data = np.array([1, 2, 3, 4, 5])
    result = array_normalization(data, 0, 1)
    assert np.allclose(result.min(), 0, atol=1e-10)
    assert np.allclose(result.max(), 1, atol=1e-10)


def test_2_gradient_edge_detector():
    """Тест 2: Градиентный детектор"""
    image = np.zeros((10, 10))
    image[:, 5:] = 255
    edges = gradient_edge_detector(image)
    assert edges.shape == image.shape
    assert edges.min() >= 0
    assert edges.max() <= 255


def test_3_gen_gray_scale_picture():
    """Тест 3: Генерация изображения"""
    result = gen_gray_scale_picture(5, 3)
    assert result.shape == (5, 5)
    assert result.min() >= 0
    assert result.max() <= 255


def test_4_sum_squared_difference():
    """Тест 4: Сумма квадратов разностей"""
    img1 = np.random.rand(10, 10)
    img2 = img1.copy()
    shift, ssd = sum_squared_difference(img1, img2)
    assert shift == (0, 0)
    assert np.allclose(ssd, 0, atol=1e-10)


def test_5_gen_discs():
    """Тест 5: Генерация дисков"""
    discs = gen_discs(3, 1)
    assert len(discs) == 1
    assert len(discs[0]) == 1
    assert discs[0][0].shape[0] == discs[0][0].shape[1]  # Квадратный


def test_6_pool_roi():
    """Тест 6: ROI pooling"""
    feature_map = np.random.rand(10, 10, 3)
    roi = [0.2, 0.2, 0.6, 0.6]
    pooled = pool_roi(feature_map, roi, 2, 2)
    assert pooled.shape == (2, 2, 3)


def test_7_pool_rois():
    """Тест 7: Множественное ROI pooling"""
    feature_map = np.random.rand(10, 10, 3)
    rois = [[0.0, 0.0, 0.5, 0.5], [0.5, 0.0, 1.0, 0.5]]
    pooled_list = pool_rois(feature_map, rois, 2, 2)
    assert len(pooled_list) == 2
    for pooled in pooled_list:
        assert pooled.shape == (2, 2, 3)


# ============================================================================
# ПАРАМЕТРИЗОВАННЫЕ ТЕСТЫ (сложный тест 1)
# ============================================================================

@pytest.mark.parametrize("input_data,range_min,range_max", [
    ([1, 2, 3, 4, 5], 0, 1),
    ([10, 20, 30], 0, 100),
    ([-5, 0, 5], -1, 1),
    ([100, 200, 300], 0, 255),
])
def test_array_normalization_parametrized(input_data, range_min, range_max):
    """Параметризованный тест нормализации"""
    result = array_normalization(np.array(input_data), range_min, range_max)
    assert np.allclose(result.min(), range_min, atol=1e-10)
    assert np.allclose(result.max(), range_max, atol=1e-10)


def test_array_normalization_all_same():
    """Тест нормализации когда все значения одинаковы"""
    data = np.array([42, 42, 42, 42], dtype=np.float64)
    result = array_normalization(data, 10, 20)
    assert np.allclose(result, 10, atol=1e-10)


# ============================================================================
# ТЕСТЫ С МОКАМИ (сложный тест 2)
# ============================================================================

@patch('perception4e_mock.np.gradient')
@patch('perception4e_mock.array_normalization')
def test_gradient_edge_detector_double_mocked(mock_normalize, mock_gradient):
    """Тест градиентного детектора с двойным моком - самый надежный вариант"""
    # Настраиваем моки
    mock_gradient.return_value = (
        np.array([[1, 2], [3, 4]], dtype=np.float64),  # gradient по x
        np.array([[5, 6], [7, 8]], dtype=np.float64)   # gradient по y
    )

    # Мок для нормализации возвращает фиксированный результат
    expected_result = np.array([[100, 150], [200, 250]], dtype=np.float64)
    mock_normalize.return_value = expected_result

    image = np.array([[0, 255], [255, 0]], dtype=np.float64)
    edges = gradient_edge_detector(image)

    # Проверяем вызовы
    assert mock_gradient.call_count == 1, "np.gradient должен быть вызван 1 раз"
    assert mock_normalize.call_count == 1, "array_normalization должен быть вызван 1 раз"

    # Проверяем аргументы вызова np.gradient
    gradient_args = mock_gradient.call_args
    assert gradient_args[0][0] is image, "np.gradient должен получить изображение"

    # Проверяем аргументы вызова array_normalization
    normalize_args = mock_normalize.call_args
    normalize_input = normalize_args[0][0]  # Входные данные для нормализации

    # Проверяем что нормализация вызывается с правильными границами
    assert normalize_args[0][1] == 0, "range_min должен быть 0"
    assert normalize_args[0][2] == 255, "range_max должен быть 255"

    # Проверяем что входные данные для нормализации - это сумма градиентов
    grad_x, grad_y = mock_gradient.return_value
    expected_input = np.abs(grad_x) + np.abs(grad_y)
    assert np.array_equal(normalize_input, expected_input), "Неправильные данные для нормализации"

    # Проверяем результат
    assert np.array_equal(edges, expected_result), "Результат должен совпадать с моком"
    assert edges.shape == image.shape, "Размеры должны совпадать"


# ============================================================================
# ДОПОЛНИТЕЛЬНЫЕ ТЕСТЫ
# ============================================================================

def test_laplacian_edge_detector():
    """Тест лапласианного детектора"""
    image = np.zeros((10, 10))
    image[:, 5:] = 255
    edges = laplacian_edge_detector(image)
    assert edges.shape == image.shape
    assert edges.min() >= 0
    assert edges.max() <= 255


def test_gaussian_derivative_edge_detector():
    """Тест гауссовского детектора"""
    image = np.random.rand(10, 10) * 255
    edges = gaussian_derivative_edge_detector(image)
    assert edges.shape == image.shape
    assert edges.min() >= 0
    assert edges.max() <= 255


def test_probability_contour_detection():
    """Тест обнаружения контуров"""
    image = np.zeros((10, 10))
    image[2:8, 2:8] = 100

    discs = gen_discs(3, 1)
    result = probability_contour_detection(image, discs[0], threshold=0)

    assert result.shape == image.shape


def test_pool_roi_with_2d_input():
    """Тест ROI pooling с 2D входом (без каналов)"""
    feature_map = np.random.rand(10, 10)
    roi = [0.2, 0.2, 0.6, 0.6]
    pooled = pool_roi(feature_map, roi, 2, 2)
    assert pooled.shape == (2, 2)


def test_pool_roi_edge_cases():
    """Тест граничных случаев ROI pooling"""
    # Случай 1: Очень маленький ROI
    feature_map = np.random.rand(10, 10, 3)
    roi = [0.9, 0.9, 0.95, 0.95]
    pooled = pool_roi(feature_map, roi, 2, 2)
    assert pooled.shape == (2, 2, 3)

    # Случай 2: Некорректный ROI
    roi_invalid = [0.8, 0.8, 0.8, 0.8]
    pooled_invalid = pool_roi(feature_map, roi_invalid, 2, 2)
    assert pooled_invalid.shape == (2, 2, 3)


def test_sum_squared_difference_with_shift():
    """Тест SSD со сдвигом"""
    img1 = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]], dtype=np.float64)

    img2 = np.roll(img1, 1, axis=0)
    img2 = np.roll(img2, 1, axis=1)

    shift, ssd = sum_squared_difference(img1, img2)

    assert isinstance(shift, tuple)
    assert len(shift) == 2
    assert isinstance(ssd, float)


# ============================================================================
# ЗАПУСК ТЕСТОВ
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Запуск тестов для perception4e_mock.py")
    print("=" * 60)

    exit_code = pytest.main([__file__, "-v", "--tb=short"])

    print("\n" + "=" * 60)
    print("📊 Статистика тестов:")
    print("=" * 60)
    print(f"✅ 7 базовых тестов")
    print(f"✅ 2 сложных теста (параметризованный и с моками)")
    print(f"✅ Всего {19} тестов")
    print("=" * 60)

    if exit_code == 0:
        print("🎉 ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
        print("✅ Лабораторная работа выполнена!")
    else:
        print("❌ Некоторые тесты не прошли")

    exit(exit_code)