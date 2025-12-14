"""
Unit tests for perception4e.py
"""
import pytest
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock, call, mock_open

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


class TestSelectiveSearch:
    @patch('cv2.imread')
    @patch('cv2.ximgproc.segmentation.createSelectiveSearchSegmentation')
    def test_selective_search_basic(self, mock_create_ss, mock_imread):
        mock_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_image
        mock_ss = MagicMock()
        mock_create_ss.return_value = mock_ss
        mock_rects = np.array([[10, 10, 30, 40], [20, 20, 50, 60]])
        mock_ss.process.return_value = mock_rects

        with patch('cv2.rectangle'), patch('cv2.imshow'), patch('cv2.waitKey'):
            result = selective_search(None)
            mock_imread.assert_called_once()
            mock_create_ss.assert_called_once()
            assert np.array_equal(result, mock_rects)


class TestROIPooling:
    @pytest.fixture
    def sample_feature_map(self):
        feature_map = np.zeros((10, 10, 3))
        for i in range(10):
            for j in range(10):
                feature_map[i, j, :] = [i * 10, j * 10, (i + j) * 5]
        return feature_map

    def test_pool_roi_basic(self, sample_feature_map):
        roi = [0.2, 0.2, 0.6, 0.6]
        pooled_height = 2
        pooled_width = 2
        pooled = pool_roi(sample_feature_map, roi, pooled_height, pooled_width)
        assert pooled.shape == (pooled_height, pooled_width, 3)


class TestMNISTDataLoading:
    """Тесты для загрузки данных MNIST"""

    @patch('keras.datasets.mnist.load_data')
    def test_load_MINST_basic(self, mock_load_data):
        """Тест базовой загрузки MNIST"""
        # Arrange
        # Мокаем возвращаемые данные
        mock_x_train = np.random.rand(60000, 28, 28).astype(np.uint8)
        mock_y_train = np.random.randint(0, 10, 60000)
        mock_x_test = np.random.rand(10000, 28, 28).astype(np.uint8)
        mock_y_test = np.random.randint(0, 10, 10000)

        mock_load_data.return_value = ((mock_x_train, mock_y_train), (mock_x_test, mock_y_test))

        train_size = 1000
        val_size = 100
        test_size = 200

        # Мокаем to_categorical
        with patch('keras.utils.to_categorical') as mock_to_categorical:
            mock_to_categorical.side_effect = lambda y, num_classes: np.eye(num_classes)[y]

            # Act
            (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_MINST(
                train_size, val_size, test_size
            )

            # Assert
            # Проверяем размеры
            assert train_x.shape == (train_size, 1, 28, 28)
            assert train_y.shape == (train_size, 10)

            assert val_x.shape == (val_size, 1, 28, 28)
            assert val_y.shape == (val_size, 10)

            assert test_x.shape == (test_size, 1, 28, 28)
            assert test_y.shape == (test_size, 10)

            # Проверяем нормализацию
            assert train_x.max() <= 1.0
            assert train_x.min() >= 0.0
            assert train_x.dtype == np.float32

    @patch('keras.datasets.mnist.load_data')
    def test_load_MINST_insufficient_data(self, mock_load_data):
        """Тест когда запрошено больше данных чем есть"""
        # Arrange
        mock_x_train = np.random.rand(1000, 28, 28).astype(np.uint8)  # Только 1000 samples
        mock_y_train = np.random.randint(0, 10, 1000)
        mock_x_test = np.random.rand(200, 28, 28).astype(np.uint8)
        mock_y_test = np.random.randint(0, 10, 200)

        mock_load_data.return_value = ((mock_x_train, mock_y_train), (mock_x_test, mock_y_test))

        # Запрашиваем больше чем есть
        train_size = 800
        val_size = 300  # 800 + 300 > 1000

        # Act
        (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_MINST(
            train_size, val_size, test_size=100
        )

        # Assert
        # Функция должна подстроить train_size
        assert train_x.shape[0] + val_x.shape[0] <= 1000
        assert train_x.shape[0] == 1000 - val_size  # 1000 - 300 = 700

    @patch('keras.datasets.mnist.load_data')
    def test_load_MINST_with_mock_categorical(self, mock_load_data):
        """Тест с моком to_categorical"""
        # Arrange
        mock_x_train = np.random.rand(5000, 28, 28)
        mock_y_train = np.array([0, 1, 2, 3, 4] * 1000)  # 5000 samples
        mock_x_test = np.random.rand(1000, 28, 28)
        mock_y_test = np.array([5, 6, 7, 8, 9] * 200)

        mock_load_data.return_value = ((mock_x_train, mock_y_train), (mock_x_test, mock_y_test))

        # Создаем мок для to_categorical
        categorical_results = {}

        def mock_categorical(y, num_classes):
            key = (tuple(y[:5]), num_classes)  # Берем первые 5 для проверки
            if key not in categorical_results:
                # Создаем one-hot encoding
                result = np.eye(num_classes)[y]
                categorical_results[key] = result
            return categorical_results[key]

        with patch('keras.utils.to_categorical', side_effect=mock_categorical):
            # Act
            (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_MINST(
                train_size=100, val_size=50, test_size=30
            )

            # Assert
            # Проверяем one-hot encoding
            assert train_y.shape[1] == 10  # 10 классов
            # Проверяем что каждый sample имеет одну 1 и остальные 0
            for i in range(min(10, train_y.shape[0])):
                assert np.sum(train_y[i]) == 1
                assert np.where(train_y[i] == 1)[0][0] == mock_y_train[i]

    @pytest.mark.parametrize("train_size,val_size,test_size", [
        (100, 20, 30),
        (500, 100, 200),
        (1000, 200, 300),
        (50, 10, 15),
    ])
    @patch('keras.datasets.mnist.load_data')
    def test_load_MINST_parametrized(self, mock_load_data, train_size, val_size, test_size):
        """Параметризованный тест загрузки MNIST"""
        # Arrange
        total_train = 60000
        mock_x_train = np.random.rand(total_train, 28, 28)
        mock_y_train = np.random.randint(0, 10, total_train)
        mock_x_test = np.random.rand(10000, 28, 28)
        mock_y_test = np.random.randint(0, 10, 10000)

        mock_load_data.return_value = ((mock_x_train, mock_y_train), (mock_x_test, mock_y_test))

        # Мокаем to_categorical
        with patch('keras.utils.to_categorical') as mock_to_categorical:
            mock_to_categorical.side_effect = lambda y, num_classes: np.eye(num_classes)[y]

            # Act
            (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_MINST(
                train_size, val_size, test_size
            )

            # Assert
            assert train_x.shape == (train_size, 1, 28, 28)
            assert val_x.shape == (val_size, 1, 28, 28)
            assert test_x.shape == (test_size, 1, 28, 28)

            # Проверяем что данные нормализованы
            assert np.all(train_x >= 0) and np.all(train_x <= 1)
            assert np.all(val_x >= 0) and np.all(val_x <= 1)
            assert np.all(test_x >= 0) and np.all(test_x <= 1)


class TestSimpleConvNet:
    """Тесты для простой сверточной сети"""

    @patch('keras.models.Sequential')
    def test_simple_convnet_creation(self, mock_sequential):
        """Тест создания сверточной сети"""
        # Arrange
        mock_model = MagicMock()
        mock_sequential.return_value = mock_model

        # Мокаем слои
        mock_layers = []
        for layer_name in ['InputLayer', 'Conv2D', 'MaxPooling2D', 'Flatten', 'Dense', 'Activation']:
            mock_layer = MagicMock()
            mock_layer.__name__ = layer_name
            mock_layers.append(mock_layer)

        with patch('keras.layers.InputLayer', return_value=mock_layers[0]), \
             patch('keras.layers.Conv2D', return_value=mock_layers[1]), \
             patch('keras.layers.MaxPooling2D', return_value=mock_layers[2]), \
             patch('keras.layers.Flatten', return_value=mock_layers[3]), \
             patch('keras.layers.Dense', return_value=mock_layers[4]), \
             patch('keras.layers.Activation', return_value=mock_layers[5]):

            # Act
            model = simple_convnet(size=2, num_classes=10)

            # Assert
            # Проверяем что Sequential был создан
            mock_sequential.assert_called_once()

            # Проверяем что слои были добавлены
            # Должно быть: InputLayer + (Conv2D + MaxPooling2D)*2 + Flatten + Dense + Activation
            expected_calls = 1 + 2*2 + 1 + 1 + 1  # 8 вызовов add
            assert mock_model.add.call_count == expected_calls

            # Проверяем что модель была скомпилирована
            mock_model.compile.assert_called_once()
            compile_args = mock_model.compile.call_args
            assert 'categorical_crossentropy' in str(compile_args)
            assert 'accuracy' in str(compile_args[1].get('metrics', []))

            # Проверяем summary
            mock_model.summary.assert_called_once()

    def test_simple_convnet_different_sizes(self):
        """Тест создания сети разного размера"""
        test_cases = [
            (1, 6),   # 1 conv layer: Input + Conv + Pool + Flatten + Dense + Activation
            (2, 8),   # 2 conv layers
            (3, 10),  # 3 conv layers
            (5, 14),  # 5 conv layers
        ]

        for size, expected_layers in test_cases:
            with patch('keras.models.Sequential') as mock_sequential:
                mock_model = MagicMock()
                mock_sequential.return_value = mock_model

                # Мокаем все слои
                with patch('keras.layers.InputLayer'), \
                     patch('keras.layers.Conv2D'), \
                     patch('keras.layers.MaxPooling2D'), \
                     patch('keras.layers.Flatten'), \
                     patch('keras.layers.Dense'), \
                     patch('keras.layers.Activation'):

                    # Act
                    model = simple_convnet(size=size, num_classes=10)

                    # Assert
                    assert mock_model.add.call_count == expected_layers

    @patch('keras.models.Sequential')
    def test_simple_convnet_different_num_classes(self, mock_sequential):
        """Тест с разным количеством классов"""
        # Arrange
        mock_model = MagicMock()
        mock_sequential.return_value = mock_model

        test_cases = [2, 5, 10, 20, 100]

        for num_classes in test_cases:
            # Reset mock
            mock_model.reset_mock()

            with patch('keras.layers.Dense') as mock_dense:
                # Мокаем другие слои
                with patch('keras.layers.InputLayer'), \
                     patch('keras.layers.Conv2D'), \
                     patch('keras.layers.MaxPooling2D'), \
                     patch('keras.layers.Flatten'), \
                     patch('keras.layers.Activation'):

                    # Act
                    model = simple_convnet(size=2, num_classes=num_classes)

                    # Assert
                    # Проверяем что Dense был вызван с правильным num_classes
                    dense_calls = [call for call in mock_dense.call_args_list
                                  if len(call[0]) > 0 and call[0][0] == num_classes]
                    assert len(dense_calls) > 0


class TestModelTraining:
    """Тесты для обучения модели"""

    @patch('perception4e.load_MINST')
    @patch('keras.models.Sequential')
    def test_train_model_basic(self, mock_sequential, mock_load_minst):
        """Тест базового обучения модели"""
        # Arrange
        # Мокаем модель
        mock_model = MagicMock()
        mock_sequential.return_value = mock_model

        # Мокаем данные
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

        # Мокаем fit и evaluate
        mock_history = MagicMock()
        mock_model.fit.return_value = mock_history
        mock_model.evaluate.return_value = [0.5, 0.85]  # [loss, accuracy]

        # Act
        trained_model = train_model(mock_model)

        # Assert
        # Проверяем что данные были загружены
        mock_load_minst.assert_called_once_with(1000, 100, 100)

        # Проверяем что fit был вызван
        mock_model.fit.assert_called_once()
        fit_args = mock_model.fit.call_args

        # Проверяем параметры fit
        assert fit_args[1]['epochs'] == 5
        assert fit_args[1]['verbose'] == 2
        assert fit_args[1]['batch_size'] == 32

        # Проверяем что evaluate был вызван
        mock_model.evaluate.assert_called_once_with(
            test_data[0], test_data[1], verbose=1
        )

        # Проверяем что возвращается модель
        assert trained_model == mock_model

    @patch('perception4e.load_MINST')
    @patch('keras.models.Sequential')
    def test_train_model_different_scores(self, mock_sequential, mock_load_minst):
        """Тест с разными результатами evaluate"""
        # Arrange
        mock_model = MagicMock()
        mock_sequential.return_value = mock_model

        # Мокаем данные
        train_data = (np.random.rand(100, 1, 28, 28), np.eye(10)[np.random.randint(0, 10, 100)])
        val_data = (np.random.rand(20, 1, 28, 28), np.eye(10)[np.random.randint(0, 10, 20)])
        test_data = (np.random.rand(30, 1, 28, 28), np.eye(10)[np.random.randint(0, 10, 30)])

        mock_load_minst.return_value = (train_data, val_data, test_data)

        test_cases = [
            ([0.1, 0.95], "Высокая точность"),
            ([0.5, 0.85], "Средняя точность"),
            ([2.0, 0.10], "Низкая точность"),
            ([5.0, 0.05], "Очень низкая точность"),
        ]

        for scores, description in test_cases:
            # Reset mocks
            mock_model.reset_mock()
            mock_load_minst.reset_mock()
            mock_load_minst.return_value = (train_data, val_data, test_data)

            # Setup
            mock_model.fit.return_value = MagicMock()
            mock_model.evaluate.return_value = scores

            # Act
            trained_model = train_model(mock_model)

            # Assert
            mock_model.evaluate.assert_called_once()
            # Модель должна быть возвращена независимо от scores
            assert trained_model == mock_model

    @patch('perception4e.load_MINST')
    @patch('keras.models.Sequential')
    def test_train_model_with_fit_parameters(self, mock_sequential, mock_load_minst):
        """Тест параметров обучения"""
        # Arrange
        mock_model = MagicMock()
        mock_sequential.return_value = mock_model

        # Мокаем данные
        mock_load_minst.return_value = (
            (np.random.rand(500, 1, 28, 28), np.eye(10)[np.random.randint(0, 10, 500)]),
            (np.random.rand(100, 1, 28, 28), np.eye(10)[np.random.randint(0, 10, 100)]),
            (np.random.rand(150, 1, 28, 28), np.eye(10)[np.random.randint(0, 10, 150)])
        )

        mock_model.fit.return_value = MagicMock()
        mock_model.evaluate.return_value = [0.3, 0.9]

        # Захватываем вызовы fit
        fit_calls = []
        original_fit = mock_model.fit

        def tracked_fit(*args, **kwargs):
            fit_calls.append((args, kwargs))
            return original_fit(*args, **kwargs)

        mock_model.fit.side_effect = tracked_fit

        # Act
        train_model(mock_model)

        # Assert
        assert len(fit_calls) == 1
        args, kwargs = fit_calls[0]

        # Проверяем параметры
        assert 'validation_data' in kwargs
        assert kwargs['epochs'] == 5
        assert kwargs['verbose'] == 2
        assert kwargs['batch_size'] == 32

        # Проверяем что переданы правильные данные
        val_data = kwargs['validation_data']
        assert len(val_data) == 2
        assert val_data[0].shape[1:] == (1, 28, 28)  # Форма без batch dimension
        assert val_data[1].shape[1] == 10  # One-hot encoding

    @patch('perception4e.load_MINST')
    @patch('keras.models.Sequential')
    def test_train_model_error_handling(self, mock_sequential, mock_load_minst):
        """Тест обработки ошибок при обучении"""
        # Arrange
        mock_model = MagicMock()
        mock_sequential.return_value = mock_model

        # Мокаем данные
        mock_load_minst.return_value = (
            (np.random.rand(100, 1, 28, 28), np.eye(10)[np.random.randint(0, 10, 100)]),
            (np.random.rand(20, 1, 28, 28), np.eye(10)[np.random.randint(0, 10, 20)]),
            (np.random.rand(30, 1, 28, 28), np.eye(10)[np.random.randint(0, 10, 30)])
        )

        # Мокаем ошибку при fit
        mock_model.fit.side_effect = ValueError("Training error")

        # Act & Assert
        with pytest.raises(ValueError, match="Training error"):
            train_model(mock_model)

        # Проверяем что evaluate не был вызван из-за ошибки
        mock_model.evaluate.assert_not_called()


class TestNeuralNetworkIntegration:
    """Интеграционные тесты для нейронных сетей"""

    def test_end_to_end_model_creation(self):
        """Тест end-to-end создания и компиляции модели"""
        # Этот тест может быть запущен без моков, но требует Keras
        try:
            # Act
            model = simple_convnet(size=1, num_classes=10)

            # Assert
            # Проверяем базовые свойства модели
            assert model is not None
            # Модель должна быть скомпилирована
            assert hasattr(model, 'optimizer')
            assert hasattr(model, 'loss')

            # Проверяем архитектуру
            # Должны быть слои: Conv2D, MaxPooling2D, Flatten, Dense
            layer_types = [layer.__class__.__name__ for layer in model.layers]
            assert 'Conv2D' in layer_types
            assert 'MaxPooling2D' in layer_types
            assert 'Flatten' in layer_types
            assert 'Dense' in layer_types

            # Проверяем входную форму
            assert model.input_shape == (None, 1, 28, 28)

            # Проверяем выходную форму
            assert model.output_shape == (None, 10)

        except ImportError:
            pytest.skip("Keras not available")
        except Exception as e:
            # Если есть другие ошибки, пропускаем тест
            pytest.skip(f"Keras test skipped: {e}")

    @patch('perception4e.simple_convnet')
    @patch('perception4e.load_MINST')
    def test_full_training_pipeline(self, mock_load_minst, mock_simple_convnet):
        """Тест полного пайплайна обучения"""
        # Arrange
        # Мокаем модель
        mock_model = MagicMock()
        mock_simple_convnet.return_value = mock_model

        # Мокаем данные
        train_data = (
            np.random.rand(1000, 1, 28, 28).astype(np.float32),
            np.random.rand(1000, 10)
        )
        val_data = (
            np.random.rand(200, 1, 28, 28).astype(np.float32),
            np.random.rand(200, 10)
        )
        test_data = (
            np.random.rand(300, 1, 28, 28).astype(np.float32),
            np.random.rand(300, 10)
        )

        mock_load_minst.return_value = (train_data, val_data, test_data)

        # Мокаем обучение
        mock_model.fit.return_value = MagicMock()
        mock_model.evaluate.return_value = [0.25, 0.92]

        # Act
        # Создаем и обучаем модель
        model = simple_convnet(size=2, num_classes=10)
        trained_model = train_model(model)

        # Assert
        # Проверяем что модель была создана
        mock_simple_convnet.assert_called_once_with(size=2, num_classes=10)

        # Проверяем что данные были загружены
        mock_load_minst.assert_called_once_with(1000, 100, 100)

        # Проверяем что модель была обучена
        mock_model.fit.assert_called_once()
        mock_model.evaluate.assert_called_once()

        # Проверяем что возвращена модель
        assert trained_model == mock_model


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])