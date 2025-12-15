"""Упрощенная версия perception4e.py для тестов"""
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
    grad_x = np.zeros_like(image)
    grad_y = np.zeros_like(image)

    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            grad_x[i, j] = image[i, j + 1] - image[i, j - 1]
            grad_y[i, j] = image[i + 1, j] - image[i - 1, j]

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
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            laplacian[i, j] = (image[i - 1, j] + image[i + 1, j] +
                               image[i, j - 1] + image[i, j + 1] - 4 * image[i, j])

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
    assert level > 0
    # init an empty image
    image = np.zeros((size, size))
    if level == 1:
        return image
    # draw a square on the left upper corner of the image
    for x in range(size):
        for y in range(size):
            image[x, y] += (250 // (level - 1)) * (max(x, y) * level // size)
    return image


def probability_contour_detection(image, discs, threshold=0):
    """
    Detect edges/contours by applying a set of discs to an image
    """
    # init an empty output image
    res = np.zeros(image.shape)
    step = discs[0].shape[0]
    for x_i in range(0, image.shape[0] - step + 1, 1):
        for y_i in range(0, image.shape[1] - step + 1, 1):
            diff = []
            # apply each pair of discs and calculate the difference
            for d in range(0, len(discs), 2):
                disc1, disc2 = discs[d], discs[d + 1]
                # crop the region of interest
                region = image[x_i: x_i + step, y_i: y_i + step]
                diff.append(np.sum(np.multiply(region, disc1)) - np.sum(np.multiply(region, disc2)))
            if max(diff) > threshold:
                # change color of the center of region
                res[x_i + step // 2, y_i + step // 2] = 255
    return res


def gen_discs(init_scale, scales=1):
    """Упрощенная генерация дисков"""
    discs = []
    for m in range(scales):
        scale = init_scale * (m + 1)
        if scale % 2 == 0:
            scale += 1

        disc = []
        # make the full empty disc
        white = np.zeros((scale, scale))
        center = (scale - 1) / 2

        for i in range(scale):
            for j in range(scale):
                if (i - center) ** 2 + (j - center) ** 2 <= (center ** 2):
                    white[i, j] = 1

        # generate halves
        half = scale // 2
        lower_half = white.copy()
        lower_half[:half, :] = 0
        upper_half = lower_half[::-1, ::-1]

        disc += [lower_half, upper_half, lower_half.T, upper_half.T]
        discs.append(disc)

    return discs


def pool_rois(feature_map, rois, pooled_height, pooled_width):
    """Упрощенный ROI pooling"""
    return [pool_roi(feature_map, roi, pooled_height, pooled_width) for roi in rois]


def pool_roi(feature_map, roi, pooled_height, pooled_width):
    """Упрощенный single ROI pooling"""
    # Ensure feature_map is at least 2D
    if len(feature_map.shape) == 2:
        h, w = feature_map.shape
        feature_map = feature_map.reshape(h, w, 1)

    h, w, c = feature_map.shape

    # Convert relative to absolute coordinates
    x1 = max(0, int(roi[0] * w))
    y1 = max(0, int(roi[1] * h))
    x2 = min(w, int(roi[2] * w))
    y2 = min(h, int(roi[3] * h))

    # Check validity
    if x2 <= x1 or y2 <= y1:
        if c == 1:
            return np.zeros((pooled_height, pooled_width))
        else:
            return np.zeros((pooled_height, pooled_width, c))

    # Simple pooling
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
                patch = feature_map[h_start:h_end, w_start:w_end, :]
                result[i, j, :] = np.max(patch, axis=(0, 1))

    # Remove last dimension if single channel
    if c == 1:
        return result[:, :, 0]

    return result


def image_to_graph(image):
    """Упрощенная конвертация изображения в граф"""
    graph_dict = {}
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            neighbors = []
            if x + 1 < image.shape[0]:
                neighbors.append((x + 1, y))
            if y + 1 < image.shape[1]:
                neighbors.append((x, y + 1))
            graph_dict[(x, y)] = neighbors
    return graph_dict


def generate_edge_weight(image, v1, v2):
    """Упрощенная генерация веса ребра"""
    diff = abs(float(image[v1[0], v1[1]]) - float(image[v2[0], v2[1]]))
    return 255.0 - diff


class Graph:
    """Упрощенный класс Graph"""

    def __init__(self, image):
        self.graph = image_to_graph(image)
        self.ROW = len(self.graph)
        self.COL = 2
        self.image = image
        self.flow = {}

        for s in self.graph:
            self.flow[s] = {}
            for t in self.graph[s]:
                if t:
                    self.flow[s][t] = generate_edge_weight(image, s, t)

    def bfs(self, s, t, parent):
        """Упрощенный BFS"""
        queue = [s]
        visited = []

        while queue:
            u = queue.pop(0)
            for node in self.graph[u]:
                if node not in visited and node and self.flow[u].get(node, 0) > 0:
                    queue.append(node)
                    visited.append(node)
                    parent.append((u, node))

        return t in visited

    def min_cut(self, source, sink):
        """Упрощенный min cut"""
        parent = []
        max_flow = 0

        while self.bfs(source, sink, parent):
            path_flow = float('inf')
            for s, t in parent:
                path_flow = min(path_flow, self.flow[s][t])

            max_flow += path_flow

            for s in self.flow:
                for t in self.flow[s]:
                    if t[0] <= sink[0] and t[1] <= sink[1]:
                        self.flow[s][t] -= path_flow
            parent = []

        res = []
        for i in self.flow:
            for j in self.flow[i]:
                if self.flow[i][j] == 0 and generate_edge_weight(self.image, i, j) > 0:
                    res.append((i, j))
        return res