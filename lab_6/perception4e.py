"""
Unit tests for perception4e.py
"""
import pytest
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

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
    Graph
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
        assert len(discs[0]) == 8


class TestSSDFunction:
    """Тесты для функции суммы квадратов разностей"""

    def test_ssd_identical_images(self):
        """Тест SSD для идентичных изображений"""
        # Arrange
        img1 = np.random.rand(10, 10)
        img2 = img1.copy()

        # Act
        shift, ssd = sum_squared_difference(img1, img2)

        # Assert
        assert shift == (0, 0)
        assert ssd == 0


class TestProbabilityContourDetection:
    """Тесты для обнаружения контуров"""

    def test_probability_contour_empty_image(self):
        """Тест обнаружения контуров на пустом изображении"""
        # Arrange
        image = np.zeros((10, 10))
        discs = gen_discs(3, 1)[0]

        # Act
        result = probability_contour_detection(image, discs, threshold=0)

        # Assert
        assert result.shape == image.shape
        assert np.all(result == 0)


class TestEdgeDetectionOperators:
    """Тесты для операторов обнаружения границ"""

    def test_gradient_edge_detector_simple_edge(self):
        """Тест градиентного детектора на простом крае"""
        # Arrange
        # Создаем изображение с вертикальным краем
        image = np.zeros((10, 10))
        image[:, 5:] = 255  # Правая половина белая

        # Act
        edges = gradient_edge_detector(image)

        # Assert
        assert edges.shape == image.shape
        assert edges.dtype in [np.float32, np.float64]
        assert edges.min() >= 0
        assert edges.max() <= 255

        # Проверяем что края обнаружены в районе x=4-6
        edge_strength = edges.sum(axis=0)  # Суммируем по вертикали
        # Ожидаем пик в районе столбца 5 (край)
        assert edge_strength[4] > edge_strength[0]  # Край сильнее чем левый край
        assert edge_strength[5] > edge_strength[9]  # Край сильнее чем правый край

    def test_gradient_edge_detector_horizontal_edge(self):
        """Тест градиентного детектора на горизонтальном крае"""
        # Arrange
        # Создаем изображение с горизонтальным краем
        image = np.zeros((10, 10))
        image[5:, :] = 255  # Нижняя половина белая

        # Act
        edges = gradient_edge_detector(image)

        # Assert
        assert edges.shape == image.shape

        # Проверяем что края обнаружены в районе y=4-6
        edge_strength = edges.sum(axis=1)  # Суммируем по горизонтали
        assert edge_strength[4] > edge_strength[0]
        assert edge_strength[5] > edge_strength[9]

    def test_gradient_edge_detector_uniform_image(self):
        """Тест на равномерном изображении (без краев)"""
        # Arrange
        image = np.ones((8, 8)) * 128  # Равномерное серое

        # Act
        edges = gradient_edge_detector(image)

        # Assert
        # На равномерном изображении не должно быть сильных краев
        assert np.all(edges >= 0)
        # Детектор может давать небольшие значения из-за численных ошибок
        assert edges.max() < 50  # Эвристический порог

    def test_gradient_edge_detector_small_image(self):
        """Тест на маленьком изображении"""
        # Arrange
        image = np.array([[0, 255], [255, 0]])  # Шахматная доска 2x2

        # Act
        edges = gradient_edge_detector(image)

        # Assert
        assert edges.shape == (2, 2)
        assert edges.min() >= 0
        assert edges.max() <= 255

    def test_gaussian_derivative_edge_detector_basic(self):
        """Тест гауссовского детектора границ"""
        # Arrange
        # Создаем тестовое изображение с краем
        image = np.zeros((15, 15))
        image[:, 8:] = 255

        # Act
        edges = gaussian_derivative_edge_detector(image)

        # Assert
        assert edges.shape == image.shape
        assert edges.dtype in [np.float32, np.float64]
        assert edges.min() >= 0
        assert edges.max() <= 255

        # Гауссовский детектор должен сглаживать, поэтому края могут быть размыты
        # но все равно должны быть обнаружены
        edge_strength = edges.sum(axis=0)
        # Проверяем что есть пик в районе края
        peak_region = edge_strength[6:10]
        assert peak_region.max() > edge_strength[:5].max()
        assert peak_region.max() > edge_strength[10:].max()

    def test_gaussian_derivative_edge_detector_noise_robustness(self):
        """Тест устойчивости гауссовского детектора к шуму"""
        # Arrange
        # Создаем изображение с краем и шумом
        image = np.zeros((20, 20))
        image[:, 10:] = 200
        # Добавляем шум
        noise = np.random.normal(0, 20, image.shape)
        noisy_image = np.clip(image + noise, 0, 255)

        # Act
        edges = gaussian_derivative_edge_detector(noisy_image)

        # Assert
        assert edges.shape == noisy_image.shape
        # Детектор должен работать даже с шумом
        assert not np.isnan(edges).any()
        assert not np.isinf(edges).any()

    def test_laplacian_edge_detector_basic(self):
        """Тест лапласианного детектора границ"""
        # Arrange
        # Создаем ступенчатый край
        image = np.zeros((10, 10))
        image[:, 5:] = 255

        # Act
        edges = laplacian_edge_detector(image)

        # Assert
        assert edges.shape == image.shape
        assert edges.dtype in [np.float32, np.float64]
        assert edges.min() >= 0
        assert edges.max() <= 255

        # Лапласиан должен давать положительные и отрицательные отклики
        # на краях, но после нормализации все значения должны быть в [0, 255]
        unique_values = np.unique(edges)
        assert len(unique_values) > 1  # Должны быть разные значения

    def test_laplacian_edge_detector_diagonal_edge(self):
        """Тест лапласианного детектора на диагональном крае"""
        # Arrange
        image = np.zeros((10, 10))
        # Создаем диагональный край
        for i in range(10):
            image[i, i:] = 255

        # Act
        edges = laplacian_edge_detector(image)

        # Assert
        assert edges.shape == image.shape
        # Проверяем что есть ненулевые значения
        assert np.any(edges > 10)  # Эвристический порог

    def test_edge_detectors_consistency(self):
        """Тест согласованности разных детекторов границ"""
        # Arrange
        # Простое изображение с краем
        image = np.zeros((8, 8))
        image[:, 4:] = 255

        # Act
        edges_grad = gradient_edge_detector(image)
        edges_gauss = gaussian_derivative_edge_detector(image)
        edges_laplacian = laplacian_edge_detector(image)

        # Assert
        # Все должны иметь одинаковую форму
        assert edges_grad.shape == edges_gauss.shape == edges_laplacian.shape

        # Все значения должны быть в диапазоне [0, 255]
        for edges in [edges_grad, edges_gauss, edges_laplacian]:
            assert edges.min() >= 0
            assert edges.max() <= 255

        # Градиентный и гауссовский детекторы должны давать похожие результаты
        # (но не идентичные из-за сглаживания)
        correlation = np.corrcoef(edges_grad.flatten(), edges_gauss.flatten())[0, 1]
        assert correlation > 0.3  # Должны быть положительно коррелированы


class TestShowEdges:
    """Тесты для функции отображения границ"""

    @patch('matplotlib.pyplot.imshow')
    @patch('matplotlib.pyplot.axis')
    @patch('matplotlib.pyplot.show')
    def test_show_edges_calls(self, mock_show, mock_axis, mock_imshow):
        """Тест что функция правильно вызывает matplotlib"""
        # Arrange
        edges = np.random.rand(10, 10) * 255

        # Act
        show_edges(edges)

        # Assert
        # Проверяем что imshow был вызван с правильными аргументами
        mock_imshow.assert_called_once()
        args, kwargs = mock_imshow.call_args
        assert np.array_equal(args[0], edges)
        assert kwargs['cmap'] == 'gray'
        assert kwargs['vmin'] == 0
        assert kwargs['vmax'] == 255

        # Проверяем что axis был вызван с 'off'
        mock_axis.assert_called_once_with('off')

        # Проверяем что show был вызван
        mock_show.assert_called_once()

    def test_show_edges_with_invalid_input(self):
        """Тест с некорректным входом"""
        # Arrange
        invalid_edges = "not an array"

        # Act & Assert
        with pytest.raises(AttributeError):
            show_edges(invalid_edges)


class TestGroupContourDetection:
    """Тесты для группового обнаружения контуров"""

    @patch('cv2.kmeans')
    def test_group_contour_detection_basic(self, mock_kmeans):
        """Тест базового обнаружения контуров"""
        # Arrange
        image = np.random.rand(10, 10) * 255
        mock_kmeans.return_value = (
            True,  # ret
            np.array([0, 1, 0, 1]).reshape(2, 2),  # label
            np.array([[100], [200]])  # center
        )

        # Act
        result = group_contour_detection(image, cluster_num=2)

        # Assert
        # Проверяем что kmeans был вызван с правильными аргументами
        mock_kmeans.assert_called_once()
        args, kwargs = mock_kmeans.call_args

        # Проверяем аргументы
        Z = args[0]
        K = args[1]
        criteria = args[3]

        assert np.array_equal(Z, np.float32(image))
        assert K == 2
        assert criteria[0] == (1 + 2)  # cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
        assert criteria[1] == 10  # max_iter
        assert criteria[2] == 1.0  # epsilon

    def test_group_contour_detection_single_cluster(self):
        """Тест с одним кластером"""
        # Arrange
        image = np.ones((5, 5)) * 128

        # Act
        result = group_contour_detection(image, cluster_num=1)

        # Assert
        assert result.shape == image.shape
        # С одним кластером все пиксели должны иметь одно значение
        unique_values = np.unique(result)
        assert len(unique_values) == 1


class TestImageGraphConversion:
    """Тесты для конвертации изображения в граф"""

    def test_image_to_graph_small(self):
        """Тест конвертации маленького изображения"""
        # Arrange
        image = np.array([[1, 2], [3, 4]])

        # Act
        graph = image_to_graph(image)

        # Assert
        assert isinstance(graph, dict)
        assert len(graph) == 4  # 2x2 = 4 вершины

        # Проверяем вершины
        expected_vertices = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for vertex in expected_vertices:
            assert vertex in graph

        # Проверяем связи для (0, 0)
        assert (1, 0) in graph[(0, 0)] or graph[(0, 0)][0] == (1, 0)
        assert (0, 1) in graph[(0, 0)] or graph[(0, 0)][1] == (0, 1)

    def test_image_to_graph_boundaries(self):
        """Тест граничных условий"""
        # Arrange
        image = np.array([[1, 2, 3]])

        # Act
        graph = image_to_graph(image)

        # Assert
        # Для (0, 2) - правый нижний угол
        # Справа None (выход за границы)
        connections = graph[(0, 2)]
        assert None in connections or len(connections) < 2

    def test_generate_edge_weight_identical(self):
        """Тест веса ребра для одинаковых пикселей"""
        # Arrange
        image = np.array([[100, 100], [100, 100]])
        v1 = (0, 0)
        v2 = (0, 1)

        # Act
        weight = generate_edge_weight(image, v1, v2)

        # Assert
        # Для одинаковых пикселей разность 0, вес = 255 - 0 = 255
        assert weight == 255

    def test_generate_edge_weight_different(self):
        """Тест веса ребра для разных пикселей"""
        # Arrange
        image = np.array([[0, 255], [0, 0]])
        v1 = (0, 0)
        v2 = (0, 1)

        # Act
        weight = generate_edge_weight(image, v1, v2)

        # Assert
        # Разность = 255, вес = 255 - 255 = 0
        assert weight == 0

    def test_generate_edge_weight_invalid_vertices(self):
        """Тест с некорректными вершинами"""
        # Arrange
        image = np.array([[1, 2], [3, 4]])

        # Act & Assert
        with pytest.raises(IndexError):
            generate_edge_weight(image, (5, 5), (0, 0))


class TestGraphClass:
    """Тесты для класса Graph"""

    def test_graph_initialization(self):
        """Тест инициализации графа"""
        # Arrange
        image = np.array([[1, 2], [3, 4]])

        # Act
        graph = Graph(image)

        # Assert
        assert graph.ROW == 4  # 2x2 = 4 вершины
        assert graph.COL == 2
        assert hasattr(graph, 'flow')
        assert hasattr(graph, 'image')

        # Проверяем flow для ребра
        assert (0, 1) in graph.flow[(0, 0)]

    def test_graph_bfs_simple(self):
        """Тест BFS на простом графе"""
        # Arrange
        image = np.array([[1, 2], [3, 4]])
        graph = Graph(image)

        # Устанавливаем нулевой flow для некоторых ребер для теста
        graph.flow[(0, 0)][(0, 1)] = 0
        graph.flow[(0, 0)][(1, 0)] = 10

        # Act
        parent = []
        result = graph.bfs((0, 0), (1, 0), parent)

        # Assert
        assert result is True
        assert len(parent) > 0

    def test_graph_bfs_no_path(self):
        """Тест BFS когда пути нет"""
        # Arrange
        image = np.array([[1, 2], [3, 4]])
        graph = Graph(image)

        # Устанавливаем нулевой flow для всех ребер
        for s in graph.flow:
            for t in graph.flow[s]:
                if t:
                    graph.flow[s][t] = 0

        # Act
        parent = []
        result = graph.bfs((0, 0), (1, 1), parent)

        # Assert
        assert result is False
        assert len(parent) == 0

    @patch.object(Graph, 'bfs')
    def test_min_cut(self, mock_bfs):
        """Тест минимального разреза"""
        # Arrange
        image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        graph = Graph(image)

        # Мокаем BFS чтобы он возвращал True первый раз и False второй
        mock_bfs.side_effect = [True, False]

        # Act
        result = graph.min_cut((0, 0), (2, 2))

        # Assert
        assert isinstance(result, list)
        # BFS должен был быть вызван 2 раза
        assert mock_bfs.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])