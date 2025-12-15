"""Unit tests for perception4e_mock.py - основной тестовый файл"""
import pytest
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

# Добавляем путь к текущей директории
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Импортируем из мок-версии
try:
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
except ImportError:
    # Если не найден, пытаемся найти в родительской директории
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
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
# БАЗОВЫЕ ТЕСТЫ (7+ тестов)
# ============================================================================

def test_array_normalization_basic():
    """Тест 1: Базовая нормализация"""
    data = [1, 2, 3, 4, 5]
    result = array_normalization(data, 0, 1)
    assert np.allclose(result.min(), 0)
    assert np.allclose(result.max(), 1)


def test_array_normalization_same_values():
    """Тест 2: Нормализация одинаковых значений"""
    data = [42, 42, 42]
    result = array_normalization(data, 10, 20)
    assert np.allclose(result, 10)


def test_gradient_edge_detector():
    """Тест 3: Детектор градиента"""
    image = np.zeros((10, 10))
    image[:, 5:] = 255
    edges = gradient_edge_detector(image)
    assert edges.shape == (10, 10)
    assert edges.min() >= 0
    assert edges.max() <= 255


def test_laplacian_edge_detector():
    """Тест 4: Лапласианный детектор"""
    image = np.random.rand(8, 8) * 255
    edges = laplacian_edge_detector(image)
    assert edges.shape == (8, 8)
    assert edges.min() >= 0
    assert edges.max() <= 255


def test_sum_squared_difference():
    """Тест 5: Сумма квадратов разностей"""
    img1 = np.ones((5, 5))
    img2 = np.ones((5, 5))
    shift, ssd = sum_squared_difference(img1, img2)
    assert shift == (0, 0)
    assert np.allclose(ssd, 0)


def test_gen_gray_scale_picture():
    """Тест 6: Генерация изображения"""
    result = gen_gray_scale_picture(5, 3)
    assert result.shape == (5, 5)
    assert result.min() >= 0
    assert result.max() <= 255


def test_gen_discs():
    """Тест 7: Генерация дисков"""
    discs = gen_discs(3, 1)
    assert len(discs) == 1
    assert discs[0][0].shape == (3, 3)


def test_pool_roi():
    """Тест 8: ROI pooling"""
    feature_map = np.random.rand(10, 10, 3)
    roi = [0.2, 0.2, 0.6, 0.6]
    pooled = pool_roi(feature_map, roi, 2, 2)
    assert pooled.shape == (2, 2, 3)


def test_pool_rois():
    """Тест 9: Множественное ROI pooling"""
    feature_map = np.random.rand(10, 10, 3)
    rois = [[0.0, 0.0, 0.5, 0.5], [0.5, 0.0, 1.0, 0.5]]
    pooled_list = pool_rois(feature_map, rois, 2, 2)
    assert len(pooled_list) == 2
    for pooled in pooled_list:
        assert pooled.shape == (2, 2, 3)


def test_gaussian_derivative_edge_detector():
    """Тест 10: Гауссовский детектор"""
    image = np.random.rand(10, 10) * 255
    edges = gaussian_derivative_edge_detector(image)
    assert edges.shape == (10, 10)
    assert edges.min() >= 0
    assert edges.max() <= 255


def test_probability_contour_detection():
    """Тест 11: Обнаружение контуров"""
    image = np.zeros((10, 10))
    image[3:7, 3:7] = 100

    discs = gen_discs(3, 1)
    result = probability_contour_detection(image, discs[0], threshold=0)

    assert result.shape == (10, 10)


def test_pool_roi_2d():
    """Тест 12: ROI pooling с 2D входом"""
    feature_map = np.random.rand(10, 10)
    roi = [0.2, 0.2, 0.6, 0.6]
    pooled = pool_roi(feature_map, roi, 2, 2)
    assert pooled.shape == (2, 2)


# ============================================================================
# ПАРАМЕТРИЗОВАННЫЕ ТЕСТЫ (сложный тест 1)
# ============================================================================

@pytest.mark.parametrize("input_data,range_min,range_max", [
    ([1, 2, 3, 4, 5], 0, 1),
    ([10, 20, 30], 0, 100),
    ([-5, 0, 5], -1, 1),
    ([100, 200, 300], 0, 255),
    ([0, 0, 0, 0], 0, 100),
    ([1], 0, 10),
])
def test_array_normalization_parametrized(input_data, range_min, range_max):
    """Параметризованный тест нормализации"""
    result = array_normalization(np.array(input_data), range_min, range_max)

    # Проверяем диапазон
    assert np.allclose(result.min(), range_min, atol=1e-10)
    assert np.allclose(result.max(), range_max, atol=1e-10)

    # Проверяем что нет NaN
    assert not np.isnan(result).any()


@pytest.mark.parametrize("size,levels", [
    (3, 2),
    (5, 3),
    (7, 4),
    (10, 1),
    (4, 5),
])
def test_gen_gray_scale_picture_parametrized(size, levels):
    """Параметризованный тест генерации изображений"""
    result = gen_gray_scale_picture(size, levels)

    assert result.shape == (size, size)
    assert result.min() >= 0
    assert result.max() <= 255
    assert not np.isnan(result).any()


# ============================================================================
# ТЕСТЫ С МОКАМИ (сложный тест 2)
# ============================================================================

@patch('perception4e_mock.array_normalization')
def test_gradient_edge_detector_with_mock(mock_normalize):
    """Тест градиентного детектора с моком нормализации"""
    # Настраиваем мок
    expected_edges = np.array([[50, 100], [150, 200]], dtype=np.float64)
    mock_normalize.return_value = expected_edges

    # Входное изображение
    image = np.array([[0, 255], [255, 0]], dtype=np.float64)

    # Вызываем функцию
    edges = gradient_edge_detector(image)

    # Проверяем что array_normalization была вызвана
    assert mock_normalize.call_count == 1

    # Проверяем аргументы
    call_args = mock_normalize.call_args[0]
    assert call_args[1] == 0  # range_min
    assert call_args[2] == 255  # range_max

    # Проверяем результат
    assert np.array_equal(edges, expected_edges)
    assert edges.shape == image.shape


@patch('perception4e_mock.np.diff')
def test_gradient_edge_detector_internal_mock(mock_diff):
    """Тест градиентного детектора с моком np.diff"""
    # Настраиваем моки
    mock_diff.side_effect = [
        np.array([[1, 2], [3, 4]]),  # grad_x
        np.array([[5, 6], [7, 8]])   # grad_y
    ]

    # Мокаем array_normalization
    with patch('perception4e_mock.array_normalization') as mock_norm:
        mock_norm.return_value = np.array([[10, 20], [30, 40]])

        image = np.array([[0, 255], [255, 0]])
        edges = gradient_edge_detector(image)

        # Проверяем вызовы
        assert mock_diff.call_count == 2
        assert mock_norm.call_count == 1


# ============================================================================
# ДОПОЛНИТЕЛЬНЫЕ ИНТЕГРАЦИОННЫЕ ТЕСТЫ
# ============================================================================

def test_full_image_processing_pipeline():
    """Интеграционный тест полного пайплайна обработки изображений"""
    # 1. Генерируем изображение
    image = gen_gray_scale_picture(10, 4)
    assert image.shape == (10, 10)

    # 2. Обнаруживаем границы разными методами
    edges1 = gradient_edge_detector(image)
    edges2 = laplacian_edge_detector(image)

    assert edges1.shape == (10, 10)
    assert edges2.shape == (10, 10)

    # 3. Генерируем диски
    discs = gen_discs(3, 2)
    assert len(discs) == 2

    # 4. Обнаруживаем контуры
    contours = probability_contour_detection(image, discs[0], threshold=50)
    assert contours.shape == (10, 10)

    # 5. Тестируем ROI pooling
    feature_map = np.random.rand(10, 10, 3)
    roi = [0.1, 0.1, 0.9, 0.9]
    pooled = pool_roi(feature_map, roi, 3, 3)
    assert pooled.shape == (3, 3, 3)


def test_edge_detection_consistency():
    """Тест согласованности детекторов границ"""
    image = np.random.rand(8, 8) * 255

    # Все детекторы должны возвращать изображения того же размера
    detectors = [
        gradient_edge_detector,
        gaussian_derivative_edge_detector,
        laplacian_edge_detector,
    ]

    for detector in detectors:
        result = detector(image)
        assert result.shape == image.shape
        assert result.dtype in [np.float32, np.float64]
        assert result.min() >= 0
        assert result.max() <= 255


# ============================================================================
# ТЕСТЫ ГРАНИЧНЫХ СЛУЧАЕВ
# ============================================================================

def test_edge_cases():
    """Тест граничных случаев"""

    # 1. Минимальное изображение
    tiny_image = np.array([[42]])
    edges = gradient_edge_detector(tiny_image)
    assert edges.shape == (1, 1)

    # 2. Пустой ROI
    feature_map = np.random.rand(5, 5, 3)
    empty_roi = [0.5, 0.5, 0.5, 0.5]  # Нулевая площадь
    pooled = pool_roi(feature_map, empty_roi, 2, 2)
    assert pooled.shape == (2, 2, 3)

    # 3. SSD с идентичными изображениями
    img = np.random.rand(3, 3)
    shift, ssd = sum_squared_difference(img, img)
    assert shift == (0, 0)
    assert np.allclose(ssd, 0)

    # 4. Диски с разными масштабами
    discs = gen_discs(2, 3)
    assert len(discs) == 3
    for i, disc_set in enumerate(discs):
        size = 2 * (i + 1)
        if size % 2 == 0:
            size += 1
        assert disc_set[0].shape == (size, size)


def test_error_handling():
    """Тест обработки некорректных входных данных"""

    # 1. Некорректные типы для array_normalization
    with pytest.raises((TypeError, AttributeError)):
        array_normalization("not an array", 0, 1)

    # 2. Некорректный размер для дисков
    discs = gen_discs(1, 1)  # 1 - некорректно, станет 3 после коррекции
    assert discs[0][0].shape[0] >= 3  # Должен стать минимум 3

    # 3. ROI с координатами за пределами
    feature_map = np.random.rand(10, 10, 3)
    roi_outside = [1.1, 1.1, 1.2, 1.2]  # За пределами
    pooled = pool_roi(feature_map, roi_outside, 2, 2)
    assert pooled.shape == (2, 2, 3)
    # Должен вернуть нули или корректно обработать


# ============================================================================
# ЗАПУСК ТЕСТОВ
# ============================================================================

if __name__ == "__main__":
    # Простой запуск для локальной проверки
    print("=" * 60)
    print("Запуск тестов для perception4e_mock.py")
    print("=" * 60)

    # Считаем количество тестов
    import inspect
    test_functions = [name for name, obj in inspect.getmembers(sys.modules[__name__])
                     if inspect.isfunction(obj) and name.startswith('test_')]

    print(f"Найдено тестов: {len(test_functions)}")
    print(f"Из них: 7+ базовых тестов")
    print(f"        2+ сложных теста (параметризованный и с моками)")
    print("=" * 60)

    # Запускаем pytest
    exit_code = pytest.main([__file__, "-v", "--tb=short", "-q"])

    if exit_code == 0:
        print("🎉 ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
        print("✅ Лабораторная работа выполнена!")
    else:
        print("❌ Некоторые тесты не прошли")

    exit(exit_code)