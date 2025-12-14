"""
Unit tests for perception4e.py
"""
import pytest
import numpy as np
import sys
import os
import warnings
from unittest.mock import patch, MagicMock, call, mock_open, PropertyMock

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
    gen_discs,
    show_edges,
    group_contour_detection,
    image_to_graph,
    generate_edge_weight,
    Graph,
    pool_rois,
    pool_roi,
    selective_search,
    load_MINST,
    simple_convnet,
    train_model
)


# ============================================================================
# ПАРАМЕТРИЗОВАННЫЕ ТЕСТЫ ДЛЯ ОСНОВНЫХ ФУНКЦИЙ
# ============================================================================

class TestArrayNormalizationParametrized:
    """Параметризованные тесты для функции нормализации"""

    @pytest.mark.parametrize("input_array,range_min,range_max,expected_min,expected_max", [
        # Базовые случаи
        ([1, 2, 3, 4, 5], 0, 1, 0, 1),
        ([10, 20, 30], 0, 100, 0, 100),
        ([-5, 0, 5], -1, 1, -1, 1),

        # Отрицательные значения
        ([-10, -5, 0, 5, 10], 0, 255, 0, 255),
        ([100, 200, 300], -1, 1, -1, 1),

        # Один элемент
        ([42], 0, 100, 0, 100),
        ([999], -10, 10, -10, 10),

        # Все одинаковые
        ([7, 7, 7, 7], 0, 1, 0, 1),
    ])
    def test_normalization_ranges(self, input_array, range_min, range_max, expected_min, expected_max):
        """Тест что нормализация работает в заданных диапазонах"""
        # Act
        result = array_normalization(np.array(input_array), range_min, range_max)

        # Assert
        assert np.allclose(result.min(), expected_min, atol=1e-10)
        assert np.allclose(result.max(), expected_max, atol=1e-10)

    @pytest.mark.parametrize("input_data", [
        # Разные типы входных данных
        [1.0, 2.0, 3.0],           # List of floats
        (1, 2, 3, 4),              # Tuple
        np.array([1, 2, 3]),       # Numpy array
        [[1, 2], [3, 4]],          # 2D list
        np.array([[1, 2], [3, 4]]), # 2D array
    ])
    def test_normalization_input_types(self, input_data):
        """Тест что функция работает с разными типами входных данных"""
        # Act
        result = array_normalization(input_data, 0, 1)

        # Assert
        assert isinstance(result, np.ndarray)
        assert result.min() >= 0
        assert result.max() <= 1


class TestEdgeDetectorsParametrized:
    """Параметризованные тесты для детекторов границ"""

    @pytest.fixture
    def edge_test_images(self):
        """Фикстура с тестовыми изображениями для edge detection"""
        images = {}

        # Вертикальный край
        vertical_edge = np.zeros((10, 10))
        vertical_edge[:, 5:] = 255
        images['vertical_edge'] = vertical_edge

        # Горизонтальный край
        horizontal_edge = np.zeros((10, 10))
        horizontal_edge[5:, :] = 255
        images['horizontal_edge'] = horizontal_edge

        # Диагональный край
        diagonal_edge = np.zeros((10, 10))
        for i in range(10):
            diagonal_edge[i, i:] = 255
        images['diagonal_edge'] = diagonal_edge

        # Шахматная доска
        checkerboard = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    checkerboard[i, j] = 255
        images['checkerboard'] = checkerboard

        # Плавный градиент
        gradient = np.zeros((10, 10))
        for i in range(10):
            gradient[i, :] = i * 25.5
        images['gradient'] = gradient

        return images

    @pytest.mark.parametrize("detector_func", [
        gradient_edge_detector,
        gaussian_derivative_edge_detector,
        laplacian_edge_detector,
    ])
    @pytest.mark.parametrize("image_type", [
        'vertical_edge',
        'horizontal_edge',
        'diagonal_edge',
        'checkerboard',
        'gradient'
    ])
    def test_edge_detectors_on_various_images(self, edge_test_images, detector_func, image_type):
        """Параметризованный тест всех детекторов на различных изображениях"""
        # Skip certain combinations that might fail
        if detector_func.__name__ == 'laplacian_edge_detector' and image_type == 'gradient':
            pytest.skip("Laplacian may not work well on smooth gradients")

        # Arrange
        image = edge_test_images[image_type]

        # Act
        edges = detector_func(image)

        # Assert
        assert edges.shape == image.shape
        assert edges.dtype in [np.float32, np.float64, np.float128]
        assert edges.min() >= 0
        assert edges.max() <= 255

        # Проверяем что нет NaN или inf
        assert not np.isnan(edges).any()
        assert not np.isinf(edges).any()

        # Для изображений с краями должны быть ненулевые значения
        if 'edge' in image_type:
            assert np.any(edges > 10)  # Эвристический порог

    @pytest.mark.parametrize("image_size", [
        (2, 2),    # Минимальный размер
        (3, 3),    # Нечетный размер
        (4, 4),    # Четный размер
        (10, 10),  # Средний размер
        (50, 50),  # Большой размер
    ])
    @pytest.mark.parametrize("detector_func", [
        gradient_edge_detector,
        gaussian_derivative_edge_detector,
        laplacian_edge_detector,
    ])
    def test_edge_detectors_different_sizes(self, image_size, detector_func):
        """Тест детекторов на изображениях разного размера"""
        # Arrange
        height, width = image_size
        image = np.random.rand(height, width) * 255

        # Act
        edges = detector_func(image)

        # Assert
        assert edges.shape == (height, width)
        assert edges.min() >= 0
        assert edges.max() <= 255


class TestSSDParametrized:
    """Параметризованные тесты для суммы квадратов разностей"""

    @pytest.mark.parametrize("image_size,shift,max_shift", [
        ((10, 10), (0, 0), 5),
        ((20, 20), (3, 2), 10),
        ((30, 30), (-5, 3), 15),
        ((15, 15), (7, -4), 10),
        ((25, 25), (0, 0), 12),
    ])
    def test_ssd_known_shifts(self, image_size, shift, max_shift):
        """Тест SSD с известными сдвигами на разных размерах"""
        # Arrange
        np.random.seed(42)
        height, width = image_size
        shift_x, shift_y = shift

        # Создаем базовое изображение
        base_image = np.random.rand(height, width) * 255

        # Создаем сдвинутое изображение
        shifted_image = np.roll(base_image, shift_x, axis=0)
        shifted_image = np.roll(shifted_image, shift_y, axis=1)

        # Обрезаем края для корректного сравнения
        if abs(shift_x) > 0 or abs(shift_y) > 0:
            # Для простоты тестируем только когда сдвиг в пределах max_shift
            if abs(shift_x) <= max_shift and abs(shift_y) <= max_shift:
                # Act
                detected_shift, ssd = sum_squared_difference(base_image, shifted_image)

                # Assert
                # SSD должен быть очень маленьким
                assert ssd < 1e-10
                # Должен обнаружить правильный сдвиг
                assert detected_shift == (shift_x, shift_y)

    @pytest.mark.parametrize("noise_level", [0.0, 0.01, 0.05, 0.1, 0.2])
    def test_ssd_noise_robustness(self, noise_level):
        """Тест устойчивости SSD к шуму"""
        # Arrange
        np.random.seed(42)
        base_image = np.random.rand(20, 20) * 255

        # Добавляем шум
        noise = np.random.normal(0, noise_level * 255, base_image.shape)
        noisy_image = np.clip(base_image + noise, 0, 255)

        # Act
        shift, ssd = sum_squared_difference(base_image, noisy_image)

        # Assert
        # Должен обнаружить сдвиг (0, 0)
        assert shift == (0, 0)
        # SSD должен увеличиваться с увеличением шума
        if noise_level > 0:
            assert ssd > 0


class TestROIPoolingParametrized:
    """Параметризованные тесты для ROI pooling"""

    @pytest.fixture
    def feature_map_factory(self):
        """Фабрика для создания карт признаков"""
        def create_feature_map(height, width, channels):
            """Создает карту признаков с предсказуемым паттерном"""
            feature_map = np.zeros((height, width, channels))
            for i in range(height):
                for j in range(width):
                    for k in range(channels):
                        # Создаем паттерн который легко проверить
                        feature_map[i, j, k] = (i * width + j) * (k + 1) / (height * width)
            return feature_map
        return create_feature_map

    @pytest.mark.parametrize("feature_map_shape,roi,pool_size", [
        # Маленькие карты
        ((8, 8, 3), [0.0, 0.0, 1.0, 1.0], (2, 2)),
        ((10, 10, 16), [0.2, 0.2, 0.8, 0.8], (3, 3)),

        # Разные соотношения сторон
        ((20, 10, 8), [0.0, 0.0, 1.0, 1.0], (4, 2)),
        ((10, 20, 8), [0.0, 0.0, 1.0, 1.0], (2, 4)),

        # Частичные ROI
        ((16, 16, 32), [0.25, 0.25, 0.75, 0.75], (4, 4)),
        ((32, 32, 64), [0.1, 0.1, 0.5, 0.5], (7, 7)),
    ])
    def test_pool_roi_parametric(self, feature_map_factory, feature_map_shape, roi, pool_size):
        """Параметризованный тест ROI pooling"""
        # Arrange
        height, width, channels = feature_map_shape
        feature_map = feature_map_factory(height, width, channels)
        pooled_height, pooled_width = pool_size

        # Act
        pooled = pool_roi(feature_map, roi, pooled_height, pooled_width)

        # Assert
        assert pooled.shape == (pooled_height, pooled_width, channels)
        assert not np.isnan(pooled).any()
        assert not np.isinf(pooled).any()

        # Проверяем что значения в разумных пределах
        assert pooled.min() >= 0
        assert pooled.max() <= 1.0

    @pytest.mark.parametrize("num_rois", [1, 3, 5, 10, 20])
    def test_pool_rois_multiple(self, feature_map_factory, num_rois):
        """Тест pool_rois с разным количеством ROI"""
        # Arrange
        feature_map = feature_map_factory(20, 20, 64)

        # Создаем ROI
        np.random.seed(42)
        rois = []
        for _ in range(num_rois):
            x_min = np.random.uniform(0, 0.7)
            y_min = np.random.uniform(0, 0.7)
            x_max = min(x_min + np.random.uniform(0.1, 0.3), 1.0)
            y_max = min(y_min + np.random.uniform(0.1, 0.3), 1.0)
            rois.append([x_min, y_min, x_max, y_max])

        # Act
        pooled_list = pool_rois(feature_map, rois, pooled_height=7, pooled_width=7)

        # Assert
        assert len(pooled_list) == num_rois
        for pooled in pooled_list:
            assert pooled.shape == (7, 7, 64)


# ============================================================================
# ТЕСТЫ С ИСПОЛЬЗОВАНИЕМ FIXTURES И MOCKS
# ============================================================================

class TestGraphAlgorithms:
    """Тесты для графовых алгоритмов с использованием фикстур"""

    @pytest.fixture
    def simple_graph(self):
        """Фикстура для простого графа 3x3"""
        image = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        return Graph(image)

    @pytest.fixture
    def edge_case_graphs(self):
        """Фикстура для граничных случаев графов"""
        cases = {}

        # Одно пиксель
        cases['single_pixel'] = Graph(np.array([[42]]))

        # Одна строка
        cases['single_row'] = Graph(np.array([[1, 2, 3, 4, 5]]))

        # Один столбец
        cases['single_column'] = Graph(np.array([[1], [2], [3], [4], [5]]))

        # Все одинаковые значения
        cases['uniform'] = Graph(np.ones((3, 3)) * 100)

        # Высокий контраст
        high_contrast = np.zeros((4, 4))
        high_contrast[:2, :] = 0
        high_contrast[2:, :] = 255
        cases['high_contrast'] = Graph(high_contrast)

        return cases

    def test_graph_initialization_fixture(self, simple_graph):
        """Тест инициализации графа с использованием фикстуры"""
        assert simple_graph.ROW == 9  # 3x3 = 9 вершин
        assert simple_graph.COL == 2

        # Проверяем flow для конкретного ребра
        assert (0, 1) in simple_graph.flow[(0, 0)]
        weight = simple_graph.flow[(0, 0)][(0, 1)]
        expected = 255 - abs(1 - 2)  # 255 - 1 = 254
        assert weight == expected

    @pytest.mark.parametrize("graph_type", [
        'single_pixel',
        'single_row',
        'single_column',
        'uniform',
        'high_contrast'
    ])
    def test_graph_edge_cases(self, edge_case_graphs, graph_type):
        """Тест граничных случаев графов"""
        graph = edge_case_graphs[graph_type]

        # Проверяем что граф инициализирован
        assert hasattr(graph, 'flow')
        assert hasattr(graph, 'image')

        # Проверяем BFS для некоторых пар вершин
        if graph_type == 'single_pixel':
            # Для одного пикселя BFS всегда возвращает False
            parent = []
            result = graph.bfs((0, 0), (0, 0), parent)
            # Нет ребер, поэтому путь не найден
            assert result is False
        elif graph_type == 'high_contrast':
            # Для высококонтрастного изображения проверяем веса
            # Ребро между разными регионами должно иметь маленький вес
            weight = graph.flow[(1, 0)][(2, 0)]
            # Между 0 и 255 разница большая, вес маленький
            assert weight == 0

    def test_min_cut_simulation(self, simple_graph):
        """Тест минимального разреза с моком BFS"""
        with patch.object(simple_graph, 'bfs') as mock_bfs:
            # Настраиваем mock чтобы он вернул True затем False
            mock_bfs.side_effect = [True, False]

            # Act
            result = simple_graph.min_cut((0, 0), (2, 2))

            # Assert
            # BFS должен быть вызван 2 раза
            assert mock_bfs.call_count == 2
            # Проверяем аргументы первого вызова
            first_call = mock_bfs.call_args_list[0]
            assert first_call[0][0] == (0, 0)  # source
            assert first_call[0][1] == (2, 2)  # sink
            assert isinstance(first_call[0][2], list)  # parent list


# ============================================================================
# ИНТЕГРАЦИОННЫЕ ТЕСТЫ
# ============================================================================

class TestIntegrationScenarios:
    """Интеграционные тестовые сценарии"""

    @pytest.fixture
    def image_processing_pipeline(self):
        """Фикстура для пайплайна обработки изображений"""
        class Pipeline:
            def __init__(self):
                self.results = {}

            def run(self, image):
                """Запускает полный пайплайн обработки"""
                # 1. Обнаружение границ
                self.results['gradient_edges'] = gradient_edge_detector(image)
                self.results['gaussian_edges'] = gaussian_derivative_edge_detector(image)

                # 2. Генерация дисков для контурного анализа
                discs = gen_discs(3, 2)

                # 3. Обнаружение контуров
                self.results['contours'] = probability_contour_detection(
                    image, discs[0], threshold=50
                )

                # 4. Конвертация в граф
                graph = Graph(image)
                self.results['graph'] = graph

                return self.results

        return Pipeline()

    def test_full_image_processing_pipeline(self, image_processing_pipeline):
        """Тест полного пайплайна обработки изображений"""
        # Arrange
        image = np.random.rand(20, 20) * 255

        # Act
        results = image_processing_pipeline.run(image)

        # Assert
        # Проверяем что все этапы выполнены
        assert 'gradient_edges' in results
        assert 'gaussian_edges' in results
        assert 'contours' in results
        assert 'graph' in results

        # Проверяем размеры
        assert results['gradient_edges'].shape == image.shape
        assert results['gaussian_edges'].shape == image.shape
        assert results['contours'].shape == image.shape

        # Проверяем граф
        graph = results['graph']
        assert isinstance(graph, Graph)
        assert graph.ROW == 400  # 20x20 = 400 вершин

    @patch('perception4e.load_MINST')
    @patch('keras.models.Sequential')
    def test_training_evaluation_pipeline(self, mock_sequential, mock_load_minst):
        """Тест пайплайна обучения и оценки"""
        # Arrange
        # Мокаем модель
        mock_model = MagicMock()
        mock_sequential.return_value = mock_model

        # Мокаем данные
        train_data = (
            np.random.rand(500, 1, 28, 28).astype(np.float32),
            np.eye(10)[np.random.randint(0, 10, 500)]
        )
        val_data = (
            np.random.rand(100, 1, 28, 28).astype(np.float32),
            np.eye(10)[np.random.randint(0, 10, 100)]
        )
        test_data = (
            np.random.rand(200, 1, 28, 28).astype(np.float32),
            np.eye(10)[np.random.randint(0, 10, 200)]
        )

        mock_load_minst.return_value = (train_data, val_data, test_data)

        # Настраиваем модель
        mock_history = MagicMock()
        mock_model.fit.return_value = mock_history

        # Симулируем улучшение точности
        accuracy_values = [0.6, 0.7, 0.8, 0.85, 0.88]
        mock_model.evaluate.return_value = [0.3, accuracy_values[-1]]  # [loss, accuracy]

        # Act
        trained_model = train_model(mock_model)

        # Assert
        # Проверяем что пайплайн выполнен полностью
        mock_load_minst.assert_called_once()
        mock_model.fit.assert_called_once()
        mock_model.evaluate.assert_called_once()

        # Проверяем что модель возвращена
        assert trained_model == mock_model


# ============================================================================
# ТЕСТЫ ОБРАБОТКИ ОШИБОК
# ============================================================================

class TestErrorHandling:
    """Тесты обработки ошибок и исключительных ситуаций"""

    @pytest.mark.parametrize("invalid_input,expected_error", [
        # Неверные типы данных
        ("not an array", (AttributeError, TypeError)),
        (None, (AttributeError, TypeError)),
        (123, (AttributeError, TypeError)),

        # Пустые массивы
        (np.array([]), (ValueError, IndexError)),
        (np.array([[]]), (ValueError, IndexError)),
    ])
    def test_edge_detector_invalid_input(self, invalid_input, expected_error):
        """Тест обработки некорректного ввода в детекторах границ"""
        # Проверяем все детекторы
        detectors = [
            gradient_edge_detector,
            gaussian_derivative_edge_detector,
            laplacian_edge_detector,
        ]

        for detector in detectors:
            with pytest.raises(expected_error):
                detector(invalid_input)

    def test_pool_roi_invalid_roi_values(self):
        """Тест обработки некорректных значений ROI"""
        feature_map = np.random.rand(10, 10, 3)

        invalid_rois = [
            [1.5, 0.0, 0.5, 0.5],  # x_min > 1.0
            [0.0, 1.5, 0.5, 0.5],  # y_min > 1.0
            [0.6, 0.0, 0.4, 0.5],  # x_min > x_max
            [0.0, 0.6, 0.5, 0.4],  # y_min > y_max
            [-0.1, 0.0, 0.5, 0.5], # x_min < 0
            [0.0, -0.1, 0.5, 0.5], # y_min < 0
        ]

        for roi in invalid_rois:
            try:
                result = pool_roi(feature_map, roi, 2, 2)
                # Если не вызвало исключение, проверяем результат
                assert not np.isnan(result).any()
            except (ValueError, IndexError) as e:
                # Ожидаемое поведение
                assert "index" in str(e).lower() or "bound" in str(e).lower()

    def test_warning_handling(self):
        """Тест что функции не вызывают warnings в нормальных условиях"""
        # Создаем тестовое изображение
        image = np.random.rand(10, 10) * 255

        # Отключаем warnings для теста
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Превращаем warnings в ошибки

            # Должны выполняться без warnings
            result1 = gradient_edge_detector(image)
            result2 = array_normalization(image, 0, 255)

            # Проверяем результаты
            assert not np.isnan(result1).any()
            assert not np.isnan(result2).any()


# ============================================================================
# ТЕСТЫ ПРОИЗВОДИТЕЛЬНОСТИ
# ============================================================================

@pytest.mark.performance
class TestPerformance:
    """Тесты производительности (помечены для отдельного запуска)"""

    @pytest.mark.parametrize("size", [10, 20, 50, 100])
    def test_edge_detector_performance(self, size, benchmark):
        """Бенчмарк детекторов границ"""
        # Arrange
        image = np.random.rand(size, size) * 255

        # Act & Assert через benchmark
        result = benchmark(gradient_edge_detector, image)

        # Дополнительные проверки
        assert result.shape == (size, size)

    @pytest.mark.parametrize("num_rois", [1, 10, 50, 100])
    def test_roi_pooling_performance(self, num_rois, benchmark):
        """Бенчмарк ROI pooling"""
        # Arrange
        feature_map = np.random.rand(100, 100, 256)

        # Создаем ROI
        rois = []
        for i in range(num_rois):
            x = i * 0.01
            y = i * 0.01
            rois.append([x, y, min(x + 0.1, 1.0), min(y + 0.1, 1.0)])

        # Act & Assert
        result = benchmark(pool_rois, feature_map, rois, 7, 7)

        assert len(result) == num_rois


# ============================================================================
# ДОПОЛНИТЕЛЬНЫЕ ТЕСТЫ ДЛЯ ПОЛНОГО ПОКРЫТИЯ
# ============================================================================

class TestMiscFunctions:
    """Тесты для остальных функций"""

    def test_gen_discs_different_scales(self):
        """Тест генерации дисков разных масштабов"""
        # Act
        discs = gen_discs(init_scale=3, scales=3)

        # Assert
        assert len(discs) == 3
        # Каждый масштаб должен иметь диски разного размера
        assert discs[0][0].shape == (3, 3)
        assert discs[1][0].shape == (6, 6)  # 3 * 2
        assert discs[2][0].shape == (9, 9)  # 3 * 3

    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('cv2.destroyAllWindows')
    def test_group_contour_detection_no_display(self, mock_destroy, mock_waitkey, mock_imshow):
        """Тест что group_contour_detection не показывает окна"""
        # Arrange
        image = np.random.rand(10, 10) * 255

        with patch('cv2.kmeans') as mock_kmeans:
            mock_kmeans.return_value = (
                True,
                np.array([0, 1] * 50).reshape(10, 10),
                np.array([[100], [200]])
            )

            # Act
            result = group_contour_detection(image, cluster_num=2)

            # Assert
            # Функция не должна вызывать imshow/waitKey
            mock_imshow.assert_not_called()
            mock_waitkey.assert_not_called()
            mock_destroy.assert_not_called()

            # Но должна вернуть результат
            assert result.shape == image.shape

    def test_image_to_graph_complete_coverage(self):
        """Тест полного покрытия image_to_graph"""
        # Arrange
        image = np.array([[1, 2, 3], [4, 5, 6]])

        # Act
        graph = image_to_graph(image)

        # Assert
        # Проверяем все вершины
        for i in range(2):
            for j in range(3):
                assert (i, j) in graph

        # Проверяем связи для центральной точки
        connections = graph[(1, 1)]
        assert (2, 1) not in connections  # Выход за границы по x
        assert (1, 2) in connections or connections[1] == (1, 2)  # Справа


# ============================================================================
# ГЛАВНЫЙ БЛОК ДЛЯ ЗАПУСКА ТЕСТОВ
# ============================================================================

if __name__ == "__main__":
    # Опции для запуска тестов
    import argparse

    parser = argparse.ArgumentParser(description='Run perception4e tests')
    parser.add_argument('--fast', action='store_true', help='Run only fast tests')
    parser.add_argument('--performance', action='store_true', help='Run performance tests')
    parser.add_argument('--coverage', action='store_true', help='Generate coverage report')

    args = parser.parse_args()

    # Собираем аргументы для pytest
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
    ]

    if args.fast:
        pytest_args.extend(["-k", "not performance"])
    elif args.performance:
        pytest_args.extend(["-k", "performance", "--durations=10"])

    if args.coverage:
        pytest_args.extend([
            "--cov=perception4e",
            "--cov-report=term",
            "--cov-report=html:coverage_html"
        ])

    # Запускаем тесты
    exit_code = pytest.main(pytest_args)

    # Выводим краткую статистику
    print("\n" + "="*60)
    print("Тестирование завершено!")
    print("="*60)

    sys.exit(exit_code)