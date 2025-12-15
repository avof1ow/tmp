"""Unit tests for perception4e_simple.py"""
import pytest
import numpy as np
import sys
import os
from unittest.mock import patch

# Добавляем путь к родительской директории
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Импортируем из упрощенной версии
from perception4e_simple import (
    array_normalization,
    gradient_edge_detector,
    gaussian_derivative_edge_detector,
    laplacian_edge_detector,
    gen_gray_scale_picture,
    sum_squared_difference,
    probability_contour_detection,
    gen_discs,
    pool_rois,
    pool_roi,
    image_to_graph,
    generate_edge_weight,
    Graph
)


# ============================================================================
# БАЗОВЫЕ ТЕСТЫ (7+ тестов)
# ============================================================================

def test_1_array_normalization():
    """Тест 1: Нормализация массива"""
    data = np.array([1, 2, 3, 4, 5])
    result = array_normalization(data, 0, 1)
    assert np.allclose(result.min(), 0, atol=1e-10)
    assert np.allclose(result.max(), 1, atol=1e-10)


def test_2_array_normalization_same_values():
    """Тест 2: Нормализация одинаковых значений"""
    data = np.array([42, 42, 42, 42], dtype=np.float64)
    result = array_normalization(data, 10, 20)
    assert np.allclose(result, 10, atol=1e-10)


def test_3_gen_gray_scale_picture():
    """Тест 3: Генерация изображения"""
    result = gen_gray_scale_picture(5, 3)
    assert result.shape == (5, 5)
    assert result.min() >= 0
    assert result.max() <= 255


def test_4_gen_discs():
    """Тест 4: Генерация дисков"""
    discs = gen_discs(3, 1)
    assert len(discs) == 1
    assert len(discs[0]) == 4  # 4 диска в упрощенной версии


def test_5_pool_roi():
    """Тест 5: ROI pooling"""
    feature_map = np.random.rand(10, 10, 3)
    roi = [0.2, 0.2, 0.6, 0.6]
    pooled = pool_roi(feature_map, roi, 2, 2)
    assert pooled.shape[0] == 2
    assert pooled.shape[1] == 2


def test_6_pool_rois():
    """Тест 6: Множественное ROI pooling"""
    feature_map = np.random.rand(10, 10, 3)
    rois = [[0.0, 0.0, 0.5, 0.5], [0.5, 0.0, 1.0, 0.5]]
    pooled_list = pool_rois(feature_map, rois, 2, 2)
    assert len(pooled_list) == 2


def test_7_sum_squared_difference():
    """Тест 7: Сумма квадратов разностей"""
    img1 = np.random.rand(5, 5)
    img2 = img1.copy()
    shift, ssd = sum_squared_difference(img1, img2)
    assert isinstance(shift, tuple)
    assert len(shift) == 2


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
    assert result.min() >= range_min - 1e-10
    assert result.max() <= range_max + 1e-10


@pytest.mark.parametrize("size,levels", [
    (3, 2),
    (5, 3),
    (7, 4),
])
def test_gen_gray_scale_picture_parametrized(size, levels):
    """Параметризованный тест генерации изображений"""
    result = gen_gray_scale_picture(size, levels)
    assert result.shape == (size, size)
    assert result.min() >= 0


# ============================================================================
# ТЕСТЫ С МОКАМИ (сложный тест 2)
# ============================================================================

@patch('perception4e_simple.np.roll')
def test_sum_squared_difference_mocked(mock_roll):
    """Тест SSD с моком"""

    # Настраиваем side_effect
    def roll_side_effect(array, shift, axis):
        return array  # Просто возвращаем тот же массив

    mock_roll.side_effect = roll_side_effect

    img1 = np.array([[1, 2], [3, 4]])
    img2 = np.array([[1, 2], [3, 4]])

    shift, ssd = sum_squared_difference(img1, img2)

    assert mock_roll.call_count > 0


@patch('perception4e_simple.array_normalization')
def test_gradient_edge_detector_mocked(mock_norm):
    """Тест градиентного детектора с моком"""
    mock_norm.return_value = np.array([[10, 20], [30, 40]])

    image = np.array([[0, 255], [255, 0]])
    edges = gradient_edge_detector(image)

    assert mock_norm.call_count == 1
    assert edges.shape == image.shape


# ============================================================================
# ДОПОЛНИТЕЛЬНЫЕ ТЕСТЫ
# ============================================================================

def test_image_to_graph():
    """Тест конвертации изображения в граф"""
    image = np.array([[1, 2], [3, 4]])
    graph = image_to_graph(image)
    assert isinstance(graph, dict)
    assert (0, 0) in graph


def test_generate_edge_weight():
    """Тест генерации веса ребра"""
    image = np.array([[10, 20], [30, 40]], dtype=np.float64)
    weight = generate_edge_weight(image, (0, 0), (0, 1))
    assert isinstance(weight, (int, float, np.integer, np.floating))
    assert weight >= 0


def test_graph_class():
    """Тест класса Graph"""
    image = np.array([[1, 2], [3, 4]])
    graph = Graph(image)
    assert hasattr(graph, 'graph')
    assert hasattr(graph, 'flow')


# ============================================================================
# ИНТЕГРАЦИОННЫЕ ТЕСТЫ
# ============================================================================

def test_edge_detection_pipeline():
    """Тест пайплайна обнаружения границ"""
    image = np.zeros((10, 10))
    image[3:7, 3:7] = 255

    detectors = [
        gradient_edge_detector,
        gaussian_derivative_edge_detector,
        laplacian_edge_detector,
    ]

    for detector in detectors:
        result = detector(image)
        assert result.shape == image.shape


def test_contour_detection_pipeline():
    """Тест пайплайна обнаружения контуров"""
    image = np.zeros((10, 10))
    image[2:8, 2:8] = 100

    discs = gen_discs(3, 1)
    result = probability_contour_detection(image, discs[0], threshold=0)
    assert result.shape == image.shape


# ============================================================================
# ЗАПУСК ТЕСТОВ
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Запуск тестов для perception4e_simple.py")
    print("=" * 60)

    exit_code = pytest.main([__file__, "-v", "--tb=short"])

    print("\n" + "=" * 60)
    print("📊 Статистика тестов:")
    print("=" * 60)
    print(f"✅ Базовые тесты: 7")
    print(f"✅ Сложные тесты: 2 (параметризованный и с моками)")
    print(f"✅ Всего тестов: 21")
    print("=" * 60)

    if exit_code == 0:
        print("🎉 ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
        print("✅ Лабораторная работа выполнена!")
    else:
        print("❌ Некоторые тесты не прошли")

    sys.exit(exit_code)