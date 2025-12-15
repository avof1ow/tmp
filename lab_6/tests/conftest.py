"""Конфигурация тестов для обработки импортов"""
import sys
import os
import numpy as np

# Добавляем путь к родительской директории
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Создаем простые моки для отсутствующих модулей
class MockModule:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        return MockModule()

# Мок для cv2
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = MockModule(
        TERM_CRITERIA_EPS=1,
        TERM_CRITERIA_MAX_ITER=2,
        KMEANS_RANDOM_CENTERS=0,
        kmeans=lambda *args, **kwargs: (None, None, None),
        imread=lambda *args, **kwargs: None,
        imshow=lambda *args, **kwargs: None,
        waitKey=lambda *args, **kwargs: -1,
        destroyAllWindows=lambda: None,
    )
    sys.modules['cv2'] = cv2

# Мок для keras
try:
    import keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    keras = MockModule(
        utils=MockModule(
            to_categorical=lambda y, num_classes=None: np.eye(num_classes)[y]
        )
    )
    sys.modules['keras'] = keras

    # Мок для keras.datasets.mnist
    mnist_module = MockModule(
        load_data=lambda: (
            (np.random.rand(60000, 28, 28), np.random.randint(0, 10, 60000)),
            (np.random.rand(10000, 28, 28), np.random.randint(0, 10, 10000))
        )
    )
    sys.modules['keras.datasets'] = MockModule(mnist=mnist_module)

    # Мок для keras.layers
    layers_module = MockModule(
        Dense=MockModule(),
        Activation=MockModule(),
        Flatten=MockModule(),
        InputLayer=MockModule(),
        Conv2D=MockModule(),
        MaxPooling2D=MockModule()
    )
    sys.modules['keras.layers'] = layers_module

    # Мок для keras.models
    models_module = MockModule(
        Sequential=type('Sequential', (), {
            '__init__': lambda self: None,
            'add': lambda self, layer: None,
            'compile': lambda self, **kwargs: None,
            'fit': lambda self, **kwargs: None,
            'evaluate': lambda self, **kwargs: [0.0, 0.0],
            'summary': lambda self: None
        })
    )
    sys.modules['keras.models'] = models_module

# Мок для scipy.signal
try:
    import scipy.signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    scipy_signal = MockModule(
        convolve2d=lambda a, b, mode: np.zeros_like(a),
        convolve=lambda a, b, mode: np.zeros_like(a)
    )
    sys.modules['scipy.signal'] = scipy_signal

# Мок для matplotlib
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    matplotlib_pyplot = MockModule(
        imshow=lambda *args, **kwargs: None,
        axis=lambda *args, **kwargs: None,
        show=lambda: None
    )
    sys.modules['matplotlib.pyplot'] = matplotlib_pyplot

# Также создаем мок для scipy модуля
if not SCIPY_AVAILABLE:
    sys.modules['scipy'] = MockModule(signal=scipy_signal)

print(f"Configuration loaded: cv2={CV2_AVAILABLE}, keras={KERAS_AVAILABLE}, "
      f"scipy={SCIPY_AVAILABLE}, matplotlib={MATPLOTLIB_AVAILABLE}")