"""
Unit tests for perception4e.py
"""
import pytest
import numpy as np
import sys
import os
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


class TestArrayNormalization:
    def test_normalization_basic(self):
        data = np.array([1, 2, 3, 4, 5])
        result = array_normalization(data, 0, 1)
        assert result.min() == 0
        assert result.max() == 1


class TestGrayScalePictureGeneration:
    def test_gen_gray_scale_picture_basic(self):
        size = 5
        level = 3
        result = gen_gray_scale_picture(size, level)
        assert result.shape == (size, size)


class TestDiscGeneration:
    def test_gen_discs_basic(self):
        init_scale = 3
        scales = 2
        discs = gen_discs(init_scale, scales)
        assert len(discs) == scales
        assert len(discs[0]) == 8


class TestSSDFunction:
    def test_ssd_identical_images(self):
        img1 = np.random.rand(10, 10)
        img2 = img1.copy()
        shift, ssd = sum_squared_difference(img1, img2)
        assert shift == (0, 0)
        assert ssd == 0


class TestProbabilityContourDetection:
    def test_probability_contour_empty_image(self):
        image = np.zeros((10, 10))
        discs = gen_discs(3, 1)[0]
        result = probability_contour_detection(image, discs, threshold=0)
        assert result.shape == image.shape
        assert np.all(result == 0)


class TestEdgeDetectionOperators:
    def test_gradient_edge_detector_simple_edge(self):
        image = np.zeros((10, 10))
        image[:, 5:] = 255
        edges = gradient_edge_detector(image)
        assert edges.shape == image.shape
        assert edges.dtype in [np.float32, np.float64]


class TestShowEdges:
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
    @patch('cv2.kmeans')
    def test_group_contour_detection_basic(self, mock_kmeans):
        image = np.random.rand(10, 10) * 255
        mock_kmeans.return_value = (True, np.array([0, 1, 0, 1]).reshape(2, 2), np.array([[100], [200]]))
        result = group_contour_detection(image, cluster_num=2)
        mock_kmeans.assert_called_once()


class TestImageGraphConversion:
    def test_image_to_graph_small(self):
        image = np.array([[1, 2], [3, 4]])
        graph = image_to_graph(image)
        assert isinstance(graph, dict)
        assert len(graph) == 4


class TestGraphClass:
    def test_graph_initialization(self):
        image = np.array([[1, 2], [3, 4]])
        graph = Graph(image)
        assert graph.ROW == 4
        assert hasattr(graph, 'flow')


class TestOpticalFlowSSD:
    @pytest.mark.parametrize("shift_x,shift_y", [
        (0, 0),
        (2, 0),
        (0, 2),
        (2, 2),
    ])
    def test_ssd_detects_known_shift(self, shift_x, shift_y):
        np.random.seed(42)
        base_image = np.random.rand(30, 30) * 255
        shifted_image = np.roll(base_image, shift_x, axis=0)
        shifted_image = np.roll(shifted_image, shift_y, axis=1)
        detected_shift, ssd = sum_squared_difference(base_image, shifted_image)
        assert ssd < 1e-10
        assert detected_shift == (shift_x, shift_y)


class TestSegmentationFunctions:
    @pytest.fixture
    def sample_image_with_regions(self):
        image = np.zeros((20, 20))
        image[2:8, 2:8] = 50
        image[2:8, 12:18] = 150
        image[12:18, 2:18] = 250
        return image

    def test_probability_contour_on_region_image(self, sample_image_with_regions):
        image = sample_image_with_regions
        discs = gen_discs(5, 1)[0]
        result = probability_contour_detection(image, discs, threshold=20)
        assert result.shape == image.shape
        contour_pixels = np.sum(result > 0)
        assert contour_pixels > 0


class TestMNISTDataLoading:
    @patch('keras.datasets.mnist.load_data')
    def test_load_MINST_basic(self, mock_load_data):
        mock_x_train = np.random.rand(60000, 28, 28).astype(np.uint8)
        mock_y_train = np.random.randint(0, 10, 60000)
        mock_x_test = np.random.rand(10000, 28, 28).astype(np.uint8)
        mock_y_test = np.random.randint(0, 10, 10000)
        mock_load_data.return_value = ((mock_x_train, mock_y_train), (mock_x_test, mock_y_test))

        with patch('keras.utils.to_categorical') as mock_to_categorical:
            mock_to_categorical.side_effect = lambda y, num_classes: np.eye(num_classes)[y]
            (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_MINST(1000, 100, 200)
            assert train_x.shape == (1000, 1, 28, 28)
            assert train_y.shape == (1000, 10)


class TestSimpleConvNet:
    @patch('keras.models.Sequential')
    def test_simple_convnet_creation(self, mock_sequential):
        mock_model = MagicMock()
        mock_sequential.return_value = mock_model

        with patch('keras.layers.InputLayer'), \
             patch('keras.layers.Conv2D'), \
             patch('keras.layers.MaxPooling2D'), \
             patch('keras.layers.Flatten'), \
             patch('keras.layers.Dense'), \
             patch('keras.layers.Activation'):

            model = simple_convnet(size=2, num_classes=10)
            mock_sequential.assert_called_once()
            mock_model.compile.assert_called_once()


class TestModelTraining:
    @patch('perception4e.load_MINST')
    @patch('keras.models.Sequential')
    def test_train_model_basic(self, mock_sequential, mock_load_minst):
        mock_model = MagicMock()
        mock_sequential.return_value = mock_model

        train_data = (
            np.random.rand(1000, 1, 28, 28).astype(np.float32),
            np.eye(10)[np.random.randint(0, 10, 1000)]
        )
        val_data = (
            np.random.rand(100, 1, 28, 28).astype(np.float32),
            np.eye(10)[np.random.randint(0, 10, 100)]
        )
        test_data = (
            np.random.rand(100, 1, 28, 28).astype(np.float32),
            np.eye(10)[np.random.randint(0, 10, 100)]
        )

        mock_load_minst.return_value = (train_data, val_data, test_data)
        mock_model.fit.return_value = MagicMock()
        mock_model.evaluate.return_value = [0.5, 0.85]

        trained_model = train_model(mock_model)
        mock_load_minst.assert_called_once_with(1000, 100, 100)
        mock_model.fit.assert_called_once()
        assert trained_model == mock_model


class TestSelectiveSearch:
    """Расширенные тесты для selective search"""

    @pytest.fixture
    def mock_cv2_environment(self):
        """Фикстура для мокинга OpenCV среды"""
        with patch('cv2.imread') as mock_imread, \
             patch('cv2.ximgproc.segmentation.createSelectiveSearchSegmentation') as mock_create_ss, \
             patch('cv2.rectangle') as mock_rectangle, \
             patch('cv2.imshow') as mock_imshow, \
             patch('cv2.waitKey') as mock_waitkey:

            # Настраиваем моки
            mock_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            mock_imread.return_value = mock_image

            mock_ss = MagicMock()
            mock_create_ss.return_value = mock_ss

            yield mock_imread, mock_create_ss, mock_ss, mock_rectangle, mock_imshow, mock_waitkey

    @pytest.mark.parametrize("rect_count", [0, 1, 5, 10, 50])
    def test_selective_search_different_rect_counts(self, mock_cv2_environment, rect_count):
        """Тест selective search с разным количеством возвращаемых прямоугольников"""
        # Arrange
        mock_imread, mock_create_ss, mock_ss, mock_rectangle, mock_imshow, mock_waitkey = mock_cv2_environment

        # Генерируем случайные прямоугольники
        mock_rects = np.array([
            [np.random.randint(0, 80), np.random.randint(0, 80),
             np.random.randint(10, 20), np.random.randint(10, 20)]
            for _ in range(rect_count)
        ])
        mock_ss.process.return_value = mock_rects

        # Act
        result = selective_search(None)

        # Assert
        assert np.array_equal(result, mock_rects)
        # rectangle должен быть вызван для каждого прямоугольника (но не более 100)
        expected_calls = min(rect_count, 100)
        assert mock_rectangle.call_count == expected_calls

    def test_selective_search_empty_result(self, mock_cv2_environment):
        """Тест когда selective search не находит регионов"""
        # Arrange
        mock_imread, mock_create_ss, mock_ss, mock_rectangle, mock_imshow, mock_waitkey = mock_cv2_environment
        mock_ss.process.return_value = np.array([])  # Пустой результат

        # Act
        result = selective_search(None)

        # Assert
        assert result.shape == (0,) or len(result) == 0
        # rectangle не должен быть вызван
        mock_rectangle.assert_not_called()

    @pytest.mark.parametrize("image_shape,expected_channels", [
        ((50, 50), 3),      # 2D -> добавляем канал
        ((50, 50, 1), 3),   # 1 канал -> преобразуем в 3
        ((50, 50, 3), 3),   # 3 канала -> оставляем как есть
        ((50, 50, 4), 3),   # 4 канала -> берем первые 3
    ])
    def test_selective_search_image_channels(self, mock_cv2_environment, image_shape, expected_channels):
        """Тест обработки изображений с разным количеством каналов"""
        # Arrange
        mock_imread, mock_create_ss, mock_ss, mock_rectangle, mock_imshow, mock_waitkey = mock_cv2_environment

        # Создаем тестовое изображение с заданной формой
        test_image = np.random.randint(0, 255, image_shape, dtype=np.uint8)
        mock_ss.process.return_value = np.array([[0, 0, 10, 10]])

        # Act
        result = selective_search(test_image)

        # Assert
        # Проверяем что изображение было преобразовано в 3 канала
        call_image = mock_ss.setBaseImage.call_args[0][0]
        assert call_image.shape[-1] == expected_channels

    def test_selective_search_rectangle_format(self, mock_cv2_environment):
        """Тест формата возвращаемых прямоугольников"""
        # Arrange
        mock_imread, mock_create_ss, mock_ss, mock_rectangle, mock_imshow, mock_waitkey = mock_cv2_environment

        # Создаем прямоугольники в формате [x, y, width, height]
        test_rects = np.array([
            [10, 20, 30, 40],   # x=10, y=20, width=30, height=40
            [50, 60, 70, 80],
        ])
        mock_ss.process.return_value = test_rects

        # Захватываем аргументы rectangle
        rectangle_calls = []
        def capture_rectangle(*args, **kwargs):
            rectangle_calls.append((args, kwargs))

        mock_rectangle.side_effect = capture_rectangle

        # Act
        result = selective_search(None)

        # Assert
        # Проверяем формат результата
        assert result.shape == (2, 4)

        # Проверяем что rectangle был вызван с правильными координатами
        assert len(rectangle_calls) == 2

        # Проверяем первый прямоугольник
        args, kwargs = rectangle_calls[0]
        # args: (image, (x, y), (x+width, y+height), color, thickness)
        assert args[1] == (10, 20)  # Начальная точка
        assert args[2] == (10 + 30, 20 + 40)  # Конечная точка (x+width, y+height)
        assert args[3] == (0, 255, 0)  # Зеленый цвет
        assert args[4] == 1  # Толщина линии

    @patch('os.path.exists', return_value=False)
    def test_selective_search_file_not_found(self, mock_exists, mock_cv2_environment):
        """Тест когда файл изображения не найден"""
        # Arrange
        mock_imread, mock_create_ss, mock_ss, mock_rectangle, mock_imshow, mock_waitkey = mock_cv2_environment
        mock_imread.return_value = None  # Имитируем отсутствие файла

        # Act & Assert
        # Функция должна обработать это (или вызвать исключение)
        try:
            result = selective_search("non_existent_image.png")
            # Если не вызвало исключение, проверяем результат
            assert result is not None
        except Exception as e:
            # Это тоже допустимое поведение
            assert "image" in str(e).lower() or "file" in str(e).lower()


class TestROIPoolingAdvanced:
    """Расширенные тесты для ROI pooling"""

    @pytest.fixture
    def sample_multi_channel_feature_map(self):
        """Фикстура для многоканальной карты признаков"""
        # Создаем карту признаков с 5 каналами
        feature_map = np.zeros((20, 20, 5))
        for i in range(20):
            for j in range(20):
                for k in range(5):
                    feature_map[i, j, k] = (i * j * (k + 1)) / 400.0
        return feature_map

    @pytest.mark.parametrize("pooled_height,pooled_width", [
        (1, 1),
        (2, 2),
        (3, 3),
        (2, 3),
        (3, 2),
        (5, 5),
        (7, 7),
    ])
    def test_pool_roi_different_pooling_sizes(self, sample_multi_channel_feature_map,
                                            pooled_height, pooled_width):
        """Тест ROI pooling с разными размерами выходного пула"""
        # Arrange
        feature_map = sample_multi_channel_feature_map
        roi = [0.1, 0.1, 0.9, 0.9]  # Центральная область

        # Act
        pooled = pool_roi(feature_map, roi, pooled_height, pooled_width)

        # Assert
        assert pooled.shape == (pooled_height, pooled_width, 5)
        # Проверяем что нет NaN значений
        assert not np.isnan(pooled).any()
        # Проверяем что значения в разумных пределах
        assert pooled.min() >= 0
        assert pooled.max() <= 1.0

    def test_pool_roi_max_operation(self):
        """Тест что pooling действительно берет максимум"""
        # Arrange
        # Создаем простую карту где мы знаем где должен быть максимум
        feature_map = np.zeros((10, 10, 2))

        # В канале 0: максимум в правом нижнем углу
        feature_map[8, 8, 0] = 100.0
        feature_map[8, 9, 0] = 90.0
        feature_map[9, 8, 0] = 80.0
        feature_map[9, 9, 0] = 70.0

        # В канале 1: максимум в левом верхнем углу
        feature_map[0, 0, 1] = 200.0
        feature_map[0, 1, 1] = 190.0
        feature_map[1, 0, 1] = 180.0
        feature_map[1, 1, 1] = 170.0

        roi = [0.0, 0.0, 1.0, 1.0]  # Вся карта
        pooled_height = 2
        pooled_width = 2

        # Act
        pooled = pool_roi(feature_map, roi, pooled_height, pooled_width)

        # Assert
        # Проверяем что в каждой ячейке пула взят максимум соответствующей области
        # Ячейка (0,0) должна содержать максимум левой верхней четверти
        assert pooled[0, 0, 0] == np.max(feature_map[:5, :5, 0])
        assert pooled[0, 0, 1] == 200.0  # Максимум канала 1

        # Ячейка (1,1) должна содержать максимум правой нижней четверти
        assert pooled[1, 1, 0] == 100.0  # Максимум канала 0
        assert pooled[1, 1, 1] == np.max(feature_map[5:, 5:, 1])

    @pytest.mark.parametrize("roi,expected_region", [
        ([0.0, 0.0, 0.5, 0.5], "top-left quarter"),
        ([0.5, 0.0, 1.0, 0.5], "top-right quarter"),
        ([0.0, 0.5, 0.5, 1.0], "bottom-left quarter"),
        ([0.5, 0.5, 1.0, 1.0], "bottom-right quarter"),
        ([0.25, 0.25, 0.75, 0.75], "center region"),
    ])
    def test_pool_roi_different_regions(self, sample_multi_channel_feature_map, roi, expected_region):
        """Тест pooling разных регионов изображения"""
        # Arrange
        feature_map = sample_multi_channel_feature_map
        height, width, channels = feature_map.shape

        # Вычисляем пиксельные координаты ROI
        h_start = int(height * roi[0])
        w_start = int(width * roi[1])
        h_end = int(height * roi[2])
        w_end = int(width * roi[3])

        # Act
        pooled = pool_roi(feature_map, roi, pooled_height=3, pooled_width=3)

        # Assert
        assert pooled.shape == (3, 3, channels)

        # Проверяем что pooled значения находятся в пределах исходной ROI
        region_values = feature_map[h_start:h_end, w_start:w_end, :]
        assert pooled.min() >= region_values.min()
        assert pooled.max() <= region_values.max()

    def test_pool_rois_empty_list(self):
        """Тест pool_rois с пустым списком ROI"""
        # Arrange
        feature_map = np.random.rand(10, 10, 3)
        rois = []  # Пустой список

        # Act
        pooled_list = pool_rois(feature_map, rois, 2, 2)

        # Assert
        assert isinstance(pooled_list, list)
        assert len(pooled_list) == 0

    def test_pool_rois_many_regions(self):
        """Тест pool_rois со многими ROI"""
        # Arrange
        feature_map = np.random.rand(20, 20, 4)

        # Создаем 10 случайных ROI
        np.random.seed(42)
        rois = []
        for _ in range(10):
            x_min = np.random.uniform(0, 0.7)
            y_min = np.random.uniform(0, 0.7)
            x_max = x_min + np.random.uniform(0.1, 0.3)
            y_max = y_min + np.random.uniform(0.1, 0.3)
            rois.append([x_min, y_min, x_max, y_max])

        # Act
        pooled_list = pool_rois(feature_map, rois, pooled_height=4, pooled_width=4)

        # Assert
        assert len(pooled_list) == 10
        for i, pooled in enumerate(pooled_list):
            assert pooled.shape == (4, 4, 4)
            # Проверяем что результаты разные для разных ROI
            if i > 0:
                assert not np.array_equal(pooled, pooled_list[i-1])

    def test_pool_roi_edge_case_small_roi(self):
        """Тест с очень маленьким ROI"""
        # Arrange
        feature_map = np.random.rand(10, 10, 3)
        # Очень маленький ROI (1 пиксель в исходных координатах)
        roi = [0.1, 0.1, 0.11, 0.11]

        # Act
        pooled = pool_roi(feature_map, roi, pooled_height=2, pooled_width=2)

        # Assert
        # Даже для маленького ROI pooling должен работать
        assert pooled.shape == (2, 2, 3)
        # Все значения должны быть одинаковы (так как pooling одного пикселя)
        assert np.allclose(pooled, pooled[0, 0])

    def test_pool_roi_with_integer_roi(self):
        """Тест с ROI в целых числах вместо дробей"""
        # Arrange
        feature_map = np.random.rand(10, 10, 3)
        # Некорректный ROI (целые числа вместо дробей)
        roi = [0, 0, 5, 5]  # Это вызовет ошибку при конвертации в int

        # Act & Assert
        # Функция должна либо сработать, либо вызвать понятную ошибку
        try:
            pooled = pool_roi(feature_map, roi, 2, 2)
            # Если сработало, проверяем результат
            assert pooled.shape == (2, 2, 3)
        except (IndexError, ValueError) as e:
            # Это ожидаемое поведение для некорректных ROI
            assert "index" in str(e).lower() or "out of bounds" in str(e).lower()


class TestObjectDetectionIntegration:
    """Интеграционные тесты для обнаружения объектов"""

    def test_pool_roi_consistency(self):
        """Тест согласованности pool_roi и pool_rois"""
        # Arrange
        feature_map = np.random.rand(15, 15, 4)
        rois = [
            [0.2, 0.2, 0.8, 0.8],
            [0.0, 0.0, 0.5, 0.5],
        ]

        # Act
        # Получаем pooled через pool_rois
        pooled_list = pool_rois(feature_map, rois, pooled_height=3, pooled_width=3)

        # Получаем pooled индивидуально через pool_roi
        pooled_individual = [
            pool_roi(feature_map, rois[0], 3, 3),
            pool_roi(feature_map, rois[1], 3, 3),
        ]

        # Assert
        assert len(pooled_list) == len(pooled_individual)
        for i in range(len(rois)):
            # Результаты должны быть одинаковыми
            assert np.array_equal(pooled_list[i], pooled_individual[i])

    @patch('perception4e.pool_roi')
    def test_pool_rois_calls_pool_roi(self, mock_pool_roi):
        """Тест что pool_rois вызывает pool_roi для каждого ROI"""
        # Arrange
        feature_map = np.random.rand(10, 10, 3)
        rois = [
            [0.0, 0.0, 0.5, 0.5],
            [0.5, 0.0, 1.0, 0.5],
            [0.0, 0.5, 0.5, 1.0],
            [0.5, 0.5, 1.0, 1.0],
        ]

        # Настраиваем mock
        mock_results = [np.random.rand(2, 2, 3) for _ in range(len(rois))]
        mock_pool_roi.side_effect = mock_results

        # Act
        pooled_list = pool_rois(feature_map, rois, pooled_height=2, pooled_width=2)

        # Assert
        # Проверяем что pool_roi был вызван для каждого ROI
        assert mock_pool_roi.call_count == len(rois)

        # Проверяем аргументы вызовов
        for i, roi in enumerate(rois):
            call_args = mock_pool_roi.call_args_list[i]
            assert np.array_equal(call_args[0][0], feature_map)
            assert call_args[0][1] == roi
            assert call_args[0][2] == 2  # pooled_height
            assert call_args[0][3] == 2  # pooled_width

        # Проверяем что возвращены правильные результаты
        assert len(pooled_list) == len(mock_results)
        for i in range(len(pooled_list)):
            assert np.array_equal(pooled_list[i], mock_results[i])

    def test_selective_search_and_pooling_integration(self):
        """Интеграционный тест selective search и ROI pooling"""
        # Этот тест требует моков для selective search
        with patch('cv2.imread') as mock_imread, \
             patch('cv2.ximgproc.segmentation.createSelectiveSearchSegmentation') as mock_create_ss, \
             patch('cv2.rectangle'), \
             patch('cv2.imshow'), \
             patch('cv2.waitKey'):

            # Arrange
            # Мокаем selective search
            mock_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            mock_imread.return_value = mock_image

            mock_ss = MagicMock()
            mock_create_ss.return_value = mock_ss

            # Создаем тестовые прямоугольники selective search
            # Формат: [x, y, width, height]
            ss_rects = np.array([
                [10, 10, 30, 40],   # ROI 1
                [50, 20, 20, 30],   # ROI 2
                [80, 80, 15, 15],   # ROI 3
            ])
            mock_ss.process.return_value = ss_rects

            # Создаем фиктивную карту признаков
            feature_map = np.random.rand(100, 100, 64)  # Больше каналов для реалистичности

            # Act
            # 1. Получаем регионы от selective search
            regions = selective_search(None)

            # 2. Конвертируем прямоугольники в формат ROI (нормализованные координаты)
            rois = []
            for rect in regions[:2]:  # Берем только первые 2 для теста
                x, y, w, h = rect
                # Конвертируем в нормализованный формат [x_min, y_min, x_max, y_max]
                roi = [
                    x / 100.0,           # x_min (нормализованный)
                    y / 100.0,           # y_min (нормализованный)
                    (x + w) / 100.0,     # x_max (нормализованный)
                    (y + h) / 100.0      # y_max (нормализованный)
                ]
                rois.append(roi)

            # 3. Применяем ROI pooling
            pooled_features = pool_rois(feature_map, rois, pooled_height=7, pooled_width=7)

            # Assert
            # Проверяем что selective search вернул регионы
            assert len(regions) == 3

            # Проверяем конвертацию ROI
            assert len(rois) == 2
            for roi in rois:
                assert 0 <= roi[0] <= 1  # x_min в [0, 1]
                assert 0 <= roi[1] <= 1  # y_min в [0, 1]
                assert roi[0] < roi[2] <= 1  # x_min < x_max <= 1
                assert roi[1] < roi[3] <= 1  # y_min < y_max <= 1

            # Проверяем ROI pooling
            assert len(pooled_features) == 2
            for pooled in pooled_features:
                assert pooled.shape == (7, 7, 64)


class TestPerformanceAndEdgeCases:
    """Тесты производительности и граничных случаев"""

    @pytest.mark.parametrize("feature_map_shape", [
        (10, 10, 3),      # Маленькая
        (50, 50, 64),     # Средняя
        (100, 100, 256),  # Большая (типичная для CNN)
        (224, 224, 512),  # Очень большая (ResNet/VGG)
    ])
    def test_pool_roi_performance(self, feature_map_shape):
        """Тест что pool_roi работает с разными размерами карт признаков"""
        # Arrange
        height, width, channels = feature_map_shape
        feature_map = np.random.rand(height, width, channels)
        roi = [0.2, 0.2, 0.8, 0.8]  # Центральная область

        # Act
        pooled = pool_roi(feature_map, roi, pooled_height=7, pooled_width=7)

        # Assert
        assert pooled.shape == (7, 7, channels)
        # Проверяем что нет NaN или inf
        assert not np.isnan(pooled).any()
        assert not np.isinf(pooled).any()

    def test_large_number_of_rois(self):
        """Тест с большим количеством ROI"""
        # Arrange
        feature_map = np.random.rand(40, 40, 128)

        # Создаем 100 ROI (типичный случай для object detection)
        np.random.seed(42)
        rois = []
        for _ in range(100):
            x_min = np.random.uniform(0, 0.8)
            y_min = np.random.uniform(0, 0.8)
            x_max = min(x_min + np.random.uniform(0.1, 0.3), 1.0)
            y_max = min(y_min + np.random.uniform(0.1, 0.3), 1.0)
            rois.append([x_min, y_min, x_max, y_max])

        # Act
        pooled_list = pool_rois(feature_map, rois, pooled_height=7, pooled_width=7)

        # Assert
        assert len(pooled_list) == 100
        # Проверяем что все результаты имеют правильную форму
        for pooled in pooled_list:
            assert pooled.shape == (7, 7, 128)

    def test_pool_roi_with_extreme_roi(self):
        """Тест с экстремальными значениями ROI"""
        # Arrange
        feature_map = np.random.rand(20, 20, 3)

        test_cases = [
            ([0.0, 0.0, 1.0, 1.0], "full image"),
            ([0.0, 0.0, 0.001, 0.001], "tiny roi"),
            ([0.999, 0.999, 1.0, 1.0], "corner pixel"),
        ]

        for roi, description in test_cases:
            # Act
            pooled = pool_roi(feature_map, roi, pooled_height=2, pooled_width=2)

            # Assert
            assert pooled.shape == (2, 2, 3), f"Failed for {description}"
            assert not np.isnan(pooled).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--durations=10"])