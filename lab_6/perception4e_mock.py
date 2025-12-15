"""Mock version of perception4e.py for testing - без зависимостей"""

import numpy as np

# ============================================================================
# Вспомогательные функции
# ============================================================================

def array_normalization(array, range_min, range_max):
    """Normalize an array in the range of (range_min, range_max)"""
    if not isinstance(array, np.ndarray):
        array = np.asarray(array, dtype=np.float64)

    array_min = np.min(array)
    array_max = np.max(array)

    # Обработка случая, когда все значения одинаковы
    if array_max == array_min:
        return np.full_like(array, range_min, dtype=np.float64)

    # Нормализация
    normalized = (array - array_min) * (range_max - range_min) / (array_max - array_min)
    return normalized + range_min


def gradient_edge_detector(image):
    """Простой детектор границ"""
    if not isinstance(image, np.ndarray):
        image = np.asarray(image, dtype=np.float64)

    # Простая имитация градиента
    grad_x = np.diff(image, axis=1, append=image[:, -1:])
    grad_y = np.diff(image, axis=0, append=image[-1:, :])
    edges = np.abs(grad_x) + np.abs(grad_y)
    return array_normalization(edges, 0, 255)


def gaussian_derivative_edge_detector(image):
    """Простой детектор границ"""
    return gradient_edge_detector(image)


def laplacian_edge_detector(image):
    """Простой детектор границ с лапласианом"""
    if not isinstance(image, np.ndarray):
        image = np.asarray(image, dtype=np.float64)

    # Простая имитация лапласиана
    laplacian = np.zeros_like(image)
    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            laplacian[i, j] = (image[i-1, j] + image[i+1, j] +
                               image[i, j-1] + image[i, j+1] - 4*image[i, j])

    edges = np.abs(laplacian)
    return array_normalization(edges, 0, 255)


def sum_squared_difference(pic1, pic2):
    """SSD of two frames - упрощенная версия"""
    pic1 = np.asarray(pic1, dtype=np.float64)
    pic2 = np.asarray(pic2, dtype=np.float64)

    min_ssd = np.inf
    min_dxy = (0, 0)

    # Простой поиск в маленьком диапазоне
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            shifted = np.roll(pic2, dx, axis=0)
            shifted = np.roll(shifted, dy, axis=1)
            ssd = np.sum((pic1 - shifted) ** 2)
            if ssd < min_ssd:
                min_ssd = ssd
                min_dxy = (dx, dy)

    return min_dxy, min_ssd


def gen_gray_scale_picture(size, level=3):
    """Простая генерация изображения"""
    image = np.zeros((size, size))
    if level <= 1:
        return image

    step = 255 / (level - 1)
    for i in range(size):
        for j in range(size):
            value = (i + j) * step / (2 * size)
            image[i, j] = min(255, value)

    return image


def probability_contour_detection(image, discs, threshold=0):
    """Упрощенное обнаружение контуров"""
    result = np.zeros_like(image)
    if not discs:
        return result

    # Используем первый диск для простоты
    disc = discs[0]
    size = disc.shape[0]

    for i in range(0, image.shape[0] - size + 1):
        for j in range(0, image.shape[1] - size + 1):
            region = image[i:i+size, j:j+size]
            response = np.sum(region * disc)
            if response > threshold:
                result[i + size//2, j + size//2] = 255

    return result


def gen_discs(init_scale, scales=1):
    """Генерация простых дисков"""
    discs = []
    for m in range(scales):
        size = init_scale * (m + 1)
        if size % 2 == 0:
            size += 1

        # Создаем простой диск
        disc = np.zeros((size, size))
        center = size // 2
        radius = center

        for i in range(size):
            for j in range(size):
                if (i - center) ** 2 + (j - center) ** 2 <= radius ** 2:
                    disc[i, j] = 1

        discs.append([disc])

    return discs


def pool_rois(feature_map, rois, pooled_height, pooled_width):
    """Упрощенный ROI pooling"""
    return [pool_roi(feature_map, roi, pooled_height, pooled_width) for roi in rois]


def pool_roi(feature_map, roi, pooled_height, pooled_width):
    """Упрощенный single ROI pooling"""
    # Конвертируем в 3D если нужно
    if len(feature_map.shape) == 2:
        h, w = feature_map.shape
        feature_map_3d = feature_map.reshape(h, w, 1)
    else:
        feature_map_3d = feature_map

    h, w, c = feature_map_3d.shape

    # Преобразуем относительные координаты в абсолютные
    x1 = max(0, int(roi[0] * w))
    y1 = max(0, int(roi[1] * h))
    x2 = min(w, int(roi[2] * w))
    y2 = min(h, int(roi[3] * h))

    # Проверяем корректность
    if x2 <= x1 or y2 <= y1:
        # Возвращаем нули
        if c == 1:
            return np.zeros((pooled_height, pooled_width))
        else:
            return np.zeros((pooled_height, pooled_width, c))

    # Вырезаем регион
    region = feature_map_3d[y1:y2, x1:x2, :]

    # Простой pooling
    result = np.zeros((pooled_height, pooled_width, c))
    h_step = max(1, (y2 - y1) // pooled_height)
    w_step = max(1, (x2 - x1) // pooled_width)

    for i in range(pooled_height):
        for j in range(pooled_width):
            h_start = y1 + i * h_step
            h_end = min(y2, y1 + (i + 1) * h_step)
            w_start = x1 + j * w_step
            w_end = min(x2, x1 + (j + 1) * w_step)

            if h_end > h_start and w_end > w_start:
                patch = region[h_start-y1:h_end-y1, w_start-x1:w_end-x1, :]
                result[i, j] = np.mean(patch, axis=(0, 1))

    # Убираем последнюю размерность если был 1 канал
    if c == 1:
        return result[:, :, 0]

    return result