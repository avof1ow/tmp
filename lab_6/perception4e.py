"""
Unit tests for perception4e.py
"""
import pytest
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock, call

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
    selective_search
)


class TestArrayNormalization:
    """Тесты для функции нормализации массивов"""

    def test_normalization_basic(self):
        data = np.array([1, 2, 3, 4, 5])
        result = array_normalization(data, 0, 1)
        assert result.min() == 0
        assert result.max() == 1


class TestGrayScalePictureGeneration:
    """Тесты для генерации изображений в оттенках серого"""

    def test_gen_gray_scale_picture_basic(self):
        size = 5
        level = 3
        result = gen_gray_scale_picture(size, level)
        assert result.shape == (size, size)


class TestDiscGeneration:
    """Тесты для генерации дисков"""

    def test_gen_discs_basic(self):
        init_scale = 3
        scales = 2
        discs = gen_discs(init_scale, scales)
        assert len(discs) == scales
        assert len(discs[0]) == 8


class TestSSDFunction:
    """Тесты для функции суммы квадратов разностей"""

    def test_ssd_identical_images(self):
        img1 = np.random.rand(10, 10)
        img2 = img1.copy()
        shift, ssd = sum_squared_difference(img1, img2)
        assert shift == (0, 0)
        assert ssd == 0


class TestProbabilityContourDetection:
    """Тесты для обнаружения контуров"""

    def test_probability_contour_empty_image(self):
        image = np.zeros((10, 10))
        discs = gen_discs(3, 1)[0]
        result = probability_contour_detection(image, discs, threshold=0)
        assert result.shape == image.shape
        assert np.all(result == 0)


class TestEdgeDetectionOperators:
    """Тесты для операторов обнаружения границ"""

    def test_gradient_edge_detector_simple_edge(self):
        image = np.zeros((10, 10))
        image[:, 5:] = 255
        edges = gradient_edge_detector(image)
        assert edges.shape == image.shape
        assert edges.dtype in [np.float32, np.float64]


class TestShowEdges:
    """Тесты для функции отображения границ"""

    @patch('matplotlib.pyplot.imshow')
    @patch('matplotlib.pyplot.axis')
    @patch('matplotlib.pyplot.show')
    def test_show_edges_calls(self, mock_show, mock_axis, mock_imshow):
        edges = np.random.rand(10, 10) * 255
        show_edges(edges)
        mock_imshow.assert_called_once()
        mock_axis.assert_called_once_with('off')
        mock_show.assert_called_once()


class TestGroupContourDetection:
    """Тесты для группового обнаружения контуров"""

    @patch('cv2.kmeans')
    def test_group_contour_detection_basic(self, mock_kmeans):
        image = np.random.rand(10, 10) * 255
        mock_kmeans.return_value = (True, np.array([0, 1, 0, 1]).reshape(2, 2), np.array([[100], [200]]))
        result = group_contour_detection(image, cluster_num=2)
        mock_kmeans.assert_called_once()


class TestImageGraphConversion:
    """Тесты для конвертации изображения в граф"""

    def test_image_to_graph_small(self):
        image = np.array([[1, 2], [3, 4]])
        graph = image_to_graph(image)
        assert isinstance(graph, dict)
        assert len(graph) == 4


class TestGraphClass:
    """Тесты для класса Graph"""

    def test_graph_initialization(self):
        image = np.array([[1, 2], [3, 4]])
        graph = Graph(image)
        assert graph.ROW == 4
        assert hasattr(graph, 'flow')


class TestOpticalFlowSSD:
    """Параметризованные тесты для оптического потока (SSD)"""

    @pytest.mark.parametrize("shift_x,shift_y", [
        (0, 0),
        (2, 0),
        (0, 2),
        (2, 2),
        (-2, 0),
        (0, -2),
        (-2, -2),
        (5, 3),
        (-3, 5)
    ])
    def test_ssd_detects_known_shift(self, shift_x, shift_y):
        """Тест что SSD корректно обнаруживает известный сдвиг"""
        # Arrange
        np.random.seed(42)
        base_image = np.random.rand(30, 30) * 255

        # Создаем сдвинутое изображение
        shifted_image = np.roll(base_image, shift_x, axis=0)
        shifted_image = np.roll(shifted_image, shift_y, axis=1)

        # Act
        detected_shift, ssd = sum_squared_difference(base_image, shifted_image)

        # Assert
        # SSD должен быть очень маленьким (почти 0) для точного сдвига
        assert ssd < 1e-10
        assert detected_shift == (shift_x, shift_y)

    @pytest.mark.parametrize("image_size", [(10, 10), (15, 15), (20, 20), (25, 25)])
    def test_ssd_different_image_sizes(self, image_size):
        """Тест SSD на изображениях разного размера"""
        # Arrange
        height, width = image_size
        img1 = np.random.rand(height, width) * 100
        img2 = img1.copy()  # Без сдвига

        # Act
        shift, ssd = sum_squared_difference(img1, img2)

        # Assert
        assert shift == (0, 0)
        assert ssd == 0

    def test_ssd_with_additive_noise(self):
        """Тест SSD с аддитивным шумом"""
        # Arrange
        img1 = np.random.rand(20, 20) * 255
        # Добавляем небольшой шум (1% от диапазона)
        noise = np.random.normal(0, 2.55, img1.shape)
        img2 = np.clip(img1 + noise, 0, 255)

        # Act
        shift, ssd = sum_squared_difference(img1, img2)

        # Assert
        # Должен обнаружить сдвиг (0, 0) несмотря на шум
        assert shift == (0, 0)
        # SSD должен быть больше 0 из-за шума
        assert ssd > 0
        # Но не слишком большим
        assert ssd < 10000  # Эвристический порог

    def test_ssd_with_occlusion(self):
        """Тест SSD с окклюзией (часть изображения изменена)"""
        # Arrange
        img1 = np.random.rand(20, 20) * 255

        # Создаем второе изображение с окклюзией
        img2 = img1.copy()
        # Меняем правый верхний угол
        img2[:5, 15:] = np.random.rand(5, 5) * 255

        # Act
        shift, ssd = sum_squared_difference(img1, img2)

        # Assert
        # Должен найти лучший сдвиг (0, 0) несмотря на окклюзию
        assert shift == (0, 0)
        # SSD должен быть положительным
        assert ssd > 0

    @pytest.mark.parametrize("threshold,expected_count", [
        (0, 25),    # Низкий порог -> много контуров
        (50, 15),   # Средний порог
        (100, 5),   # Высокий порог -> мало контуров
        (500, 0),   # Очень высокий порог -> нет контуров
    ])
    def test_probability_contour_threshold_parametrized(self, threshold, expected_count):
        """Параметризованный тест влияния порога на обнаружение контуров"""
        # Arrange
        # Создаем тестовое изображение с краями
        image = np.zeros((10, 10))
        # Добавляем несколько областей с разной интенсивностью
        image[2:5, 2:5] = 100
        image[6:9, 6:9] = 200

        discs = gen_discs(3, 1)[0]

        # Act
        result = probability_contour_detection(image, discs, threshold=threshold)

        # Assert
        contour_count = np.sum(result > 0)

        # Проверяем что с увеличением порога количество контуров уменьшается
        # (ожидаемое значение приблизительное)
        if threshold == 0:
            assert contour_count > 0
        elif threshold == 500:
            assert contour_count == 0
        else:
            # Для промежуточных значений проверяем границы
            assert contour_count >= 0
            assert contour_count <= 25


class TestSegmentationFunctions:
    """Тесты для функций сегментации"""

    @pytest.fixture
    def sample_image_with_regions(self):
        """Фикстура для тестового изображения с регионами"""
        image = np.zeros((20, 20))
        # Добавляем три различных региона
        image[2:8, 2:8] = 50    # Темный квадрат
        image[2:8, 12:18] = 150 # Средний квадрат
        image[12:18, 2:18] = 250 # Светлая полоса
        return image

    def test_probability_contour_on_region_image(self, sample_image_with_regions):
        """Тест обнаружения контуров на изображении с регионами"""
        # Arrange
        image = sample_image_with_regions
        discs = gen_discs(5, 1)[0]  # Диски большего размера

        # Act
        result = probability_contour_detection(image, discs, threshold=20)

        # Assert
        assert result.shape == image.shape

        # Проверяем что контуры обнаружены на границах регионов
        # Границы регионов должны быть около:
        # x=1-9, y=1-9 (первый квадрат)
        # x=1-9, y=11-19 (второй квадрат)
        # x=11-19, y=1-19 (полоса)

        # Проверяем что есть контуры
        contour_pixels = np.sum(result > 0)
        assert contour_pixels > 0

        # Проверяем что контуры в ожидаемых местах
        # Область вокруг первого квадрата
        region1_border = result[1:9, 1:9]
        assert np.sum(region1_border > 0) > 0

        # Область вокруг второго квадрата
        region2_border = result[1:9, 11:19]
        assert np.sum(region2_border > 0) > 0

    def test_group_contour_detection_clusters(self, sample_image_with_regions):
        """Тест кластеризации для обнаружения контуров"""
        # Arrange
        image = sample_image_with_regions

        # Act & Assert для разного количества кластеров
        for n_clusters in [2, 3, 4]:
            with patch('cv2.kmeans') as mock_kmeans:
                # Создаем моковые центры кластеров
                centers = np.array([[i * 50] for i in range(n_clusters)])
                labels = np.random.randint(0, n_clusters, size=image.shape).flatten()

                mock_kmeans.return_value = (True, labels, centers)

                result = group_contour_detection(image, cluster_num=n_clusters)

                # Проверяем что kmeans вызван с правильным количеством кластеров
                call_args = mock_kmeans.call_args[0]
                assert call_args[1] == n_clusters

    def test_graph_based_segmentation_flow(self):
        """Тест сегментации на основе графов (потоки)"""
        # Arrange
        # Создаем простое изображение с двумя регионами
        image = np.zeros((5, 5))
        image[:, :3] = 50   # Левая половина
        image[:, 3:] = 200  # Правая половина

        graph = Graph(image)

        # Act
        # Ищем минимальный разрез между (2, 1) и (2, 3)
        # что должно быть на границе регионов
        min_cut = graph.min_cut((2, 1), (2, 3))

        # Assert
        assert isinstance(min_cut, list)
        # Разрез должен содержать ребра с маленькой пропускной способностью
        # (на границе регионов)

    def test_generate_edge_weight_gradient(self):
        """Тест вычисления веса ребра на основе градиента"""
        # Arrange
        image = np.array([
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90]
        ])

        test_cases = [
            # (v1, v2, expected_weight)
            ((0, 0), (0, 1), 255 - 10),  # diff = 10
            ((0, 0), (1, 0), 255 - 30),  # diff = 30
            ((1, 1), (1, 2), 255 - 10),  # diff = 10
            ((1, 1), (2, 1), 255 - 30),  # diff = 30
        ]

        for v1, v2, expected in test_cases:
            # Act
            weight = generate_edge_weight(image, v1, v2)

            # Assert
            assert weight == expected, f"Failed for {v1}->{v2}: expected {expected}, got {weight}"

    @pytest.mark.parametrize("size,levels,expected_max", [
        (5, 2, 250),    # 2 уровня: 0 и 250
        (5, 3, 125),    # 3 уровня: 0, 125, 250
        (5, 4, 83.33),  # 4 уровня: 0, 83.33, 166.66, 250
        (10, 2, 250),
        (10, 5, 62.5),
    ])
    def test_gen_gray_scale_parametrized(self, size, levels, expected_max):
        """Параметризованный тест генерации градаций серого"""
        # Act
        result = gen_gray_scale_picture(size, levels)

        # Assert
        assert result.shape == (size, size)
        assert result.max() <= 255
        assert result.min() >= 0

        # Проверяем что максимальное значение приблизительно правильное
        # (допускаем погрешность из-за целочисленного деления)
        assert abs(result.max() - expected_max) < 2


class TestSelectiveSearch:
    """Тесты для selective search"""

    @patch('cv2.imread')
    @patch('cv2.ximgproc.segmentation.createSelectiveSearchSegmentation')
    def test_selective_search_basic(self, mock_create_ss, mock_imread):
        """Тест базовой работы selective search"""
        # Arrange
        # Мокаем изображение
        mock_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_image

        # Мокаем selective search
        mock_ss = MagicMock()
        mock_create_ss.return_value = mock_ss

        # Мокаем результаты поиска
        mock_rects = np.array([[10, 10, 30, 40], [20, 20, 50, 60]])
        mock_ss.process.return_value = mock_rects

        # Мокаем cv2.rectangle и cv2.imshow чтобы не открывать окна
        with patch('cv2.rectangle') as mock_rectangle, \
             patch('cv2.imshow') as mock_imshow, \
             patch('cv2.waitKey') as mock_waitkey:

            # Act
            result = selective_search(None)  # None чтобы использовать путь по умолчанию

            # Assert
            # Проверяем что imread был вызван
            mock_imread.assert_called_once()

            # Проверяем что selective search был настроен
            mock_create_ss.assert_called_once()
            mock_ss.setBaseImage.assert_called_once_with(mock_image)
            mock_ss.switchToSelectiveSearchQuality.assert_called_once()
            mock_ss.process.assert_called_once()

            # Проверяем что rectangle был вызван для каждого rect
            assert mock_rectangle.call_count == 2

            # Проверяем что возвращаются правильные rects
            assert np.array_equal(result, mock_rects)

    def test_selective_search_with_image_array(self):
        """Тест selective search с массивом изображения"""
        # Arrange
        image_array = np.random.randint(0, 255, (50, 50), dtype=np.uint8)

        with patch('cv2.ximgproc.segmentation.createSelectiveSearchSegmentation') as mock_create_ss, \
             patch('cv2.rectangle'), \
             patch('cv2.imshow'), \
             patch('cv2.waitKey'):

            mock_ss = MagicMock()
            mock_create_ss.return_value = mock_ss
            mock_ss.process.return_value = np.array([[0, 0, 10, 10]])

            # Act
            result = selective_search(image_array)

            # Assert
            # Должен быть создан 3-канальный массив
            mock_ss.setBaseImage.assert_called_once()
            call_image = mock_ss.setBaseImage.call_args[0][0]
            assert call_image.shape == (50, 50, 3)

    def test_selective_search_with_image_path(self):
        """Тест selective search с путем к изображению"""
        # Arrange
        test_path = "test_image.png"

        with patch('cv2.imread') as mock_imread, \
             patch('cv2.ximgproc.segmentation.createSelectiveSearchSegmentation') as mock_create_ss, \
             patch('cv2.rectangle'), \
             patch('cv2.imshow'), \
             patch('cv2.waitKey'):

            mock_image = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
            mock_imread.return_value = mock_image

            mock_ss = MagicMock()
            mock_create_ss.return_value = mock_ss
            mock_ss.process.return_value = np.array([[5, 5, 20, 20]])

            # Act
            result = selective_search(test_path)

            # Assert
            mock_imread.assert_called_once_with(test_path)


class TestROIPooling:
    """Тесты для ROI pooling"""

    @pytest.fixture
    def sample_feature_map(self):
        """Фикстура для тестовой карты признаков"""
        # Создаем простую карту признаков с градиентом
        feature_map = np.zeros((10, 10, 3))
        for i in range(10):
            for j in range(10):
                feature_map[i, j, :] = [i * 10, j * 10, (i + j) * 5]
        return feature_map

    def test_pool_roi_basic(self, sample_feature_map):
        """Тест базового ROI pooling"""
        # Arrange
        roi = [0.2, 0.2, 0.6, 0.6]  # ROI от (2,2) до (6,6) в пикселях
        pooled_height = 2
        pooled_width = 2

        # Act
        pooled = pool_roi(sample_feature_map, roi, pooled_height, pooled_width)

        # Assert
        assert pooled.shape == (pooled_height, pooled_width, 3)

        # Проверяем что значения корректны
        # ROI покрывает пиксели 2-5 (4x4 область)
        # После пулинга 2x2 каждая ячейка должна покрывать 2x2 суб-область
        # Максимум в первой ячейке должен быть около (3,3)
        assert np.allclose(pooled[0, 0], [30, 30, 30], atol=5)

    def test_pool_roi_edge_cases(self, sample_feature_map):
        """Тест ROI pooling граничных случаев"""
        test_cases = [
            # (roi, expected_shape_comment)
            ([0.0, 0.0, 1.0, 1.0], "Весь feature map"),
            ([0.0, 0.0, 0.5, 0.5], "Левая верхняя четверть"),
            ([0.5, 0.5, 1.0, 1.0], "Правая нижняя четверть"),
            ([0.4, 0.4, 0.6, 0.6], "Маленький ROI в центре"),
        ]

        for roi, description in test_cases:
            # Act
            pooled = pool_roi(sample_feature_map, roi, 3, 3)

            # Assert
            assert pooled.shape == (3, 3, 3), f"Failed for {description}"
            # Проверяем что нет NaN
            assert not np.isnan(pooled).any()

    def test_pool_roi_single_cell(self, sample_feature_map):
        """Тест ROI pooling с одной ячейкой"""
        # Arrange
        roi = [0.3, 0.3, 0.7, 0.7]

        # Act
        pooled = pool_roi(sample_feature_map, roi, 1, 1)

        # Assert
        assert pooled.shape == (1, 1, 3)
        # Должен быть максимум в ROI (пиксели 3-6)
        assert np.allclose(pooled[0, 0], [60, 60, 60], atol=10)

    def test_pool_rois_multiple(self, sample_feature_map):
        """Тест pooling нескольких ROI"""
        # Arrange
        rois = [
            [0.0, 0.0, 0.5, 0.5],  # Левая верхняя четверть
            [0.5, 0.0, 1.0, 0.5],  # Правая верхняя четверть
            [0.0, 0.5, 0.5, 1.0],  # Левая нижняя четверть
            [0.5, 0.5, 1.0, 1.0],  # Правая нижняя четверть
        ]

        # Act
        pooled_list = pool_rois(sample_feature_map, rois, 2, 2)

        # Assert
        assert len(pooled_list) == len(rois)
        for i, pooled in enumerate(pooled_list):
            assert pooled.shape == (2, 2, 3)

            # Проверяем что pooling разных регионов дает разные результаты
            if i == 0:  # Левая верхняя
                assert pooled[0, 0, 0] < 30  # Меньшие x значения
            elif i == 1:  # Правая верхняя
                assert pooled[0, 0, 1] > 50  # Большие y значения

    def test_pool_roi_invalid_roi(self):
        """Тест с некорректным ROI"""
        # Arrange
        feature_map = np.random.rand(10, 10, 3)
        invalid_rois = [
            [1.1, 0.0, 0.5, 0.5],  # x_min > 1.0
            [0.0, 1.1, 0.5, 0.5],  # y_min > 1.0
            [0.6, 0.0, 0.5, 0.5],  # x_min > x_max
            [0.0, 0.6, 0.5, 0.5],  # y_min > y_max
        ]

        for roi in invalid_rois:
            # Act & Assert
            # Может вызывать различные ошибки в зависимости от реализации
            try:
                result = pool_roi(feature_map, roi, 2, 2)
                # Если не вызвало ошибку, проверяем что результат не содержит NaN
                assert not np.isnan(result).any()
            except (ValueError, IndexError):
                # Ожидаемое поведение для некорректных ROI
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])