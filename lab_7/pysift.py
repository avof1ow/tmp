from numpy import (all, any, array, arctan2, cos, sin, exp, dot, log, logical_and, roll,
                   sqrt, stack, trace, pi, deg2rad, rad2deg, where, zeros, floor, round,
                   float32, int32, uint8, arange, meshgrid)
from numpy.linalg import det, lstsq, norm
from cv2 import resize, GaussianBlur, subtract, KeyPoint, INTER_LINEAR, INTER_NEAREST
from functools import cmp_to_key
import logging

####################
# Global variables #
####################

logger = logging.getLogger(__name__)
float_tolerance = 1e-7

# Предварительно вычисленные константы
SQRT_2 = sqrt(2.0)
DEG_TO_BIN = 36 / 360.0
BIN_TO_DEG = 360.0 / 36
INV_255 = 1.0 / 255.0
HIST_SMOOTH_COEFFS = array([1, 4, 6, 4, 1]) / 16.0


#################
# Main function #
#################

def computeKeypointsAndDescriptors(image, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5):
    """Compute SIFT keypoints and descriptors for an input image
    """
    image = image.astype('float32')
    base_image = generateBaseImageOptimized(image, sigma, assumed_blur)
    num_octaves = computeNumberOfOctaves(base_image.shape)
    gaussian_kernels = generateGaussianKernelsOptimized(sigma, num_intervals)
    gaussian_images = generateGaussianImagesOptimized(base_image, num_octaves, gaussian_kernels)
    dog_images = generateDoGImagesOptimized(gaussian_images)
    keypoints = findScaleSpaceExtremaOptimized(gaussian_images, dog_images, num_intervals, sigma, image_border_width)
    keypoints = removeDuplicateKeypoints(keypoints)
    keypoints = convertKeypointsToInputImageSize(keypoints)
    descriptors = generateDescriptorsOptimized(keypoints, gaussian_images)
    return keypoints, descriptors


#########################
# Image pyramid related #
#########################

def generateBaseImageOptimized(image, sigma, assumed_blur):
    """Optimized base image generation"""
    logger.debug('Generating base image...')
    image = resize(image, (0, 0), fx=2, fy=2, interpolation=INTER_LINEAR)
    sigma_diff_sq = max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01)
    sigma_diff = sqrt(sigma_diff_sq)
    return GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)


def computeNumberOfOctaves(image_shape):
    """Compute number of octaves in image pyramid"""
    return int(round(log(min(image_shape)) / log(2) - 1))


def generateGaussianKernelsOptimized(sigma, num_intervals):
    """Optimized Gaussian kernels generation"""
    logger.debug('Generating scales...')
    num_images = num_intervals + 3
    k = 2 ** (1.0 / num_intervals)
    kernels = zeros(num_images)
    kernels[0] = sigma

    # Предварительно вычисляем степени k
    k_powers = ones(num_images)
    for i in range(1, num_images):
        k_powers[i] = k_powers[i - 1] * k

    for i in range(1, num_images):
        sigma_prev = k_powers[i - 1] * sigma
        sigma_total = k * sigma_prev
        kernels[i] = sqrt(sigma_total ** 2 - sigma_prev ** 2)

    return kernels


def generateGaussianImagesOptimized(image, num_octaves, gaussian_kernels):
    """Optimized Gaussian images generation"""
    logger.debug('Generating Gaussian images...')
    gaussian_images = []
    current_image = image.copy()  # Копируем чтобы не менять оригинал

    for octave in range(num_octaves):
        octave_images = [current_image]

        # Применяем все ядра кроме первого (оно уже применено)
        for kernel in gaussian_kernels[1:]:
            current_image = GaussianBlur(current_image, (0, 0), sigmaX=kernel, sigmaY=kernel)
            octave_images.append(current_image)

        gaussian_images.append(octave_images)

        # Подготавливаем изображение для следующей октавы
        if octave < num_octaves - 1:
            octave_base = octave_images[-3]
            h, w = octave_base.shape[:2]
            current_image = resize(octave_base, (w // 2, h // 2), interpolation=INTER_NEAREST)

    return array(gaussian_images, dtype=object)


def generateDoGImagesOptimized(gaussian_images):
    """Optimized DoG images generation"""
    logger.debug('Generating Difference-of-Gaussian images...')
    dog_images = []

    for octave_images in gaussian_images:
        octave_dogs = []
        # Вычисляем разности между соседними изображениями
        for i in range(len(octave_images) - 1):
            dog = subtract(octave_images[i + 1], octave_images[i])
            octave_dogs.append(dog)
        dog_images.append(octave_dogs)

    return array(dog_images, dtype=object)


###############################
# Scale-space extrema related #
###############################

def findScaleSpaceExtremaOptimized(gaussian_images, dog_images, num_intervals, sigma, image_border_width,
                                   contrast_threshold=0.04):
    """Optimized scale-space extrema finding"""
    logger.debug('Finding scale-space extrema...')
    threshold = floor(0.5 * contrast_threshold / num_intervals * 255)
    keypoints = []

    for octave_idx, octave_dogs in enumerate(dog_images):
        if len(octave_dogs) < 3:
            continue

        height, width = octave_dogs[0].shape

        # Предварительно вычисляем границы
        row_min = image_border_width
        row_max = height - image_border_width
        col_min = image_border_width
        col_max = width - image_border_width

        # Обрабатываем только внутренние изображения
        for img_idx in range(1, len(octave_dogs) - 1):
            prev = octave_dogs[img_idx - 1]
            curr = octave_dogs[img_idx]
            next_img = octave_dogs[img_idx + 1]

            # Быстрый проход по пикселям
            for i in range(row_min, row_max):
                # Предварительно получаем строки
                i_min, i_max = i - 1, i + 2

                for j in range(col_min, col_max):
                    center = curr[i, j]

                    if abs(center) <= threshold:
                        continue

                    # Быстрая проверка экстремума
                    if not isFastExtremum(prev[i_min:i_max, j - 1:j + 2],
                                          curr[i_min:i_max, j - 1:j + 2],
                                          next_img[i_min:i_max, j - 1:j + 2],
                                          center):
                        continue

                    # Локализация экстремума
                    result = localizeExtremumFast(i, j, img_idx, octave_idx,
                                                  num_intervals, octave_dogs,
                                                  sigma, contrast_threshold,
                                                  image_border_width, threshold)

                    if result is None:
                        continue

                    keypoint, loc_img_idx = result
                    orientations = computeKeypointsWithOrientationsOptimized(
                        keypoint, octave_idx,
                        gaussian_images[octave_idx][loc_img_idx]
                    )

                    if orientations:
                        keypoints.extend(orientations)

    return keypoints


def isFastExtremum(prev_block, curr_block, next_block, center):
    """Быстрая проверка экстремума"""
    if center > 0:
        # Проверка текущего слоя
        if (center < curr_block[0, 0] or center < curr_block[0, 1] or
                center < curr_block[0, 2] or center < curr_block[1, 0] or
                center < curr_block[1, 2] or center < curr_block[2, 0] or
                center < curr_block[2, 1] or center < curr_block[2, 2]):
            return False

        # Проверка соседних слоев
        if any(center < prev_block) or any(center < next_block):
            return False

        return True
    else:
        # Проверка текущего слоя
        if (center > curr_block[0, 0] or center > curr_block[0, 1] or
                center > curr_block[0, 2] or center > curr_block[1, 0] or
                center > curr_block[1, 2] or center > curr_block[2, 0] or
                center > curr_block[2, 1] or center > curr_block[2, 2]):
            return False

        # Проверка соседних слоев
        if any(center > prev_block) or any(center > next_block):
            return False

        return True


def localizeExtremumFast(i, j, img_idx, octave_idx, num_intervals, dog_images,
                         sigma, contrast_threshold, border_width, init_threshold):
    """Быстрая локализация экстремума"""
    height, width = dog_images[0].shape

    for attempt in range(5):
        # Получаем блоки 3x3x3
        prev = dog_images[img_idx - 1]
        curr = dog_images[img_idx]
        next_img = dog_images[img_idx + 1]

        # Создаем куб
        cube = stack([
            prev[i - 1:i + 2, j - 1:j + 2],
            curr[i - 1:i + 2, j - 1:j + 2],
            next_img[i - 1:i + 2, j - 1:j + 2]
        ]).astype('float32') * INV_255

        # Вычисляем градиент и гессиан
        grad = computeGradientFast(cube)
        hess = computeHessianFast(cube)

        try:
            update = -lstsq(hess, grad, rcond=None)[0]
        except:
            return None

        # Проверка сходимости
        if all(abs(update) < 0.5):
            break

        # Обновляем координаты
        j_new = j + int(round(update[0]))
        i_new = i + int(round(update[1]))
        img_idx_new = img_idx + int(round(update[2]))

        # Проверка границ
        if (i_new < border_width or i_new >= height - border_width or
                j_new < border_width or j_new >= width - border_width or
                img_idx_new < 1 or img_idx_new > num_intervals):
            return None

        i, j, img_idx = i_new, j_new, img_idx_new

    # Проверка контраста
    center_val = cube[1, 1, 1]
    func_val = center_val + 0.5 * dot(grad, update)

    if abs(func_val) * num_intervals < contrast_threshold:
        return None

    # Проверка собственных значений
    xy_hess = hess[:2, :2]
    det_val = det(xy_hess)

    if det_val <= 0:
        return None

    trace_val = trace(xy_hess)
    if 10 * (trace_val ** 2) >= 121 * det_val:
        return None

    # Создаем ключевую точку
    keypoint = KeyPoint()
    scale_factor = 2 ** octave_idx
    keypoint.pt = ((j + update[0]) * scale_factor,
                   (i + update[1]) * scale_factor)

    scale_offset = int(round((update[2] + 0.5) * 255))
    keypoint.octave = (octave_idx & 255) | ((img_idx & 255) << 8) | ((scale_offset & 255) << 16)

    scale_exp = (img_idx + update[2]) / float32(num_intervals)
    keypoint.size = sigma * (2 ** scale_exp) * (2 ** (octave_idx + 1))

    keypoint.response = abs(func_val)

    return keypoint, img_idx


def computeGradientFast(cube):
    """Быстрое вычисление градиента"""
    dx = (cube[1, 1, 2] - cube[1, 1, 0]) * 0.5
    dy = (cube[1, 2, 1] - cube[1, 0, 1]) * 0.5
    ds = (cube[2, 1, 1] - cube[0, 1, 1]) * 0.5
    return array([dx, dy, ds])


def computeHessianFast(cube):
    """Быстрое вычисление гессиана"""
    center = cube[1, 1, 1]

    dxx = cube[1, 1, 2] - 2.0 * center + cube[1, 1, 0]
    dyy = cube[1, 2, 1] - 2.0 * center + cube[1, 0, 1]
    dss = cube[2, 1, 1] - 2.0 * center + cube[0, 1, 1]

    dxy = (cube[1, 2, 2] - cube[1, 2, 0] -
           cube[1, 0, 2] + cube[1, 0, 0]) * 0.25

    dxs = (cube[2, 1, 2] - cube[2, 1, 0] -
           cube[0, 1, 2] + cube[0, 1, 0]) * 0.25

    dys = (cube[2, 2, 1] - cube[2, 0, 1] -
           cube[0, 2, 1] + cube[0, 0, 1]) * 0.25

    return array([[dxx, dxy, dxs],
                  [dxy, dyy, dys],
                  [dxs, dys, dss]])


#########################
# Keypoint orientations #
#########################

def computeKeypointsWithOrientationsOptimized(keypoint, octave_idx, gaussian_image):
    """Optimized orientations computation"""
    image_shape = gaussian_image.shape

    # Предварительные вычисления
    scale = 1.5 * keypoint.size / float32(2 ** (octave_idx + 1))
    radius = int(round(3 * scale))
    weight_factor = -0.5 / (scale ** 2)

    base_y = int(round(keypoint.pt[1] / float32(2 ** octave_idx)))
    base_x = int(round(keypoint.pt[0] / float32(2 ** octave_idx)))

    # Определяем границы
    y_min = max(1, base_y - radius)
    y_max = min(image_shape[0] - 2, base_y + radius)
    x_min = max(1, base_x - radius)
    x_max = min(image_shape[1] - 2, base_x + radius)

    if y_min >= y_max or x_min >= x_max:
        return []

    # Вычисляем градиенты
    region = gaussian_image[y_min - 1:y_max + 2, x_min - 1:x_max + 2]
    height, width = y_max - y_min + 1, x_max - x_min + 1

    dx = region[1:height + 1, 2:width + 2] - region[1:height + 1, 0:width]
    dy = region[0:height, 1:width + 1] - region[2:height + 2, 1:width + 1]

    # Магнитуда и ориентация
    magnitude = sqrt(dx * dx + dy * dy)
    orientation = rad2deg(arctan2(dy, dx))

    # Гистограмма
    histogram = zeros(36)

    for i in range(height):
        y_offset = (y_min + i) - base_y
        for j in range(width):
            x_offset = (x_min + j) - base_x

            weight = exp(weight_factor * (y_offset ** 2 + x_offset ** 2))
            bin_idx = int(round(orientation[i, j] * DEG_TO_BIN)) % 36
            histogram[bin_idx] += weight * magnitude[i, j]

    # Сглаживание
    smoothed = zeros(36)
    for n in range(36):
        smoothed[n] = (
                HIST_SMOOTH_COEFFS[0] * histogram[(n - 2) % 36] +
                HIST_SMOOTH_COEFFS[1] * histogram[(n - 1) % 36] +
                HIST_SMOOTH_COEFFS[2] * histogram[n] +
                HIST_SMOOTH_COEFFS[3] * histogram[(n + 1) % 36] +
                HIST_SMOOTH_COEFFS[4] * histogram[(n + 2) % 36]
        )

    # Поиск пиков
    max_val = smoothed.max()
    if max_val < float_tolerance:
        return []

    left = roll(smoothed, 1)
    right = roll(smoothed, -1)
    peaks = where(logical_and(smoothed > left, smoothed > right))[0]

    keypoints = []
    threshold = 0.8 * max_val

    for peak in peaks:
        if smoothed[peak] < threshold:
            continue

        left_val = smoothed[(peak - 1) % 36]
        right_val = smoothed[(peak + 1) % 36]

        interp = (peak + 0.5 * (left_val - right_val) /
                  (left_val - 2 * smoothed[peak] + right_val)) % 36

        angle = 360.0 - interp * BIN_TO_DEG
        if abs(angle - 360.0) < float_tolerance:
            angle = 0

        new_kp = KeyPoint(*keypoint.pt, keypoint.size, angle,
                          keypoint.response, keypoint.octave)
        keypoints.append(new_kp)

    return keypoints


##############################
# Duplicate keypoint removal #
##############################

def compareKeypoints(kp1, kp2):
    """Compare keypoints for sorting"""
    if kp1.pt[0] != kp2.pt[0]:
        return kp1.pt[0] - kp2.pt[0]
    if kp1.pt[1] != kp2.pt[1]:
        return kp1.pt[1] - kp2.pt[1]
    if kp1.size != kp2.size:
        return kp2.size - kp1.size
    if kp1.angle != kp2.angle:
        return kp1.angle - kp2.angle
    if kp1.response != kp2.response:
        return kp2.response - kp1.response
    if kp1.octave != kp2.octave:
        return kp2.octave - kp1.octave
    return kp2.class_id - kp1.class_id


def removeDuplicateKeypoints(keypoints):
    """Remove duplicate keypoints"""
    if len(keypoints) < 2:
        return keypoints

    keypoints.sort(key=cmp_to_key(compareKeypoints))
    unique = [keypoints[0]]

    for kp in keypoints[1:]:
        last = unique[-1]
        if (last.pt[0] != kp.pt[0] or last.pt[1] != kp.pt[1] or
                last.size != kp.size or last.angle != kp.angle):
            unique.append(kp)

    return unique


#############################
# Keypoint scale conversion #
#############################

def convertKeypointsToInputImageSize(keypoints):
    """Convert keypoints to input image size"""
    converted = []
    for kp in keypoints:
        kp.pt = tuple(0.5 * array(kp.pt))
        kp.size *= 0.5
        kp.octave = (kp.octave & ~255) | ((kp.octave - 1) & 255)
        converted.append(kp)
    return converted


#########################
# Descriptor generation #
#########################

def unpackOctave(keypoint):
    """Unpack octave information"""
    octave = keypoint.octave & 255
    layer = (keypoint.octave >> 8) & 255

    if octave >= 128:
        octave = octave | -128

    if octave >= 0:
        scale = 1.0 / float32(1 << octave)
    else:
        scale = float32(1 << -octave)

    return octave, layer, scale


def generateDescriptorsOptimized(keypoints, gaussian_images):
    """Optimized descriptor generation"""
    descriptors = []

    for kp in keypoints:
        octave, layer, scale = unpackOctave(kp)
        img = gaussian_images[octave + 1, layer]
        h, w = img.shape

        point = (scale * array(kp.pt)).round().astype(int32)
        angle = 360.0 - kp.angle
        angle_rad = deg2rad(angle)
        cos_a = cos(angle_rad)
        sin_a = sin(angle_rad)

        # Размер окна
        hist_width = 1.5 * scale * kp.size
        half_width = int(min(
            round(hist_width * SQRT_2 * 2.5),  # (4+1)*0.5 = 2.5
            sqrt(h * h + w * w)
        ))

        # Границы
        r_min = max(1, point[1] - half_width)
        r_max = min(h - 2, point[1] + half_width)
        c_min = max(1, point[0] - half_width)
        c_max = min(w - 2, point[0] + half_width)

        if r_min >= r_max or c_min >= c_max:
            descriptors.append(zeros(128, dtype='float32'))
            continue

        # Создаем дескриптор
        rows = arange(r_min, r_max + 1) - point[1]
        cols = arange(c_min, c_max + 1) - point[0]
        col_grid, row_grid = meshgrid(cols, rows)

        # Вращение
        row_rot = col_grid * sin_a + row_grid * cos_a
        col_rot = col_grid * cos_a - row_grid * sin_a

        # Бин координаты
        row_bin = (row_rot / hist_width) + 1.5  # 0.5*4 - 0.5 = 1.5
        col_bin = (col_rot / hist_width) + 1.5

        # Маска
        mask = (row_bin > -1) & (row_bin < 4) & (col_bin > -1) & (col_bin < 4)

        if not mask.any():
            descriptors.append(zeros(128, dtype='float32'))
            continue

        # Получаем валидные пиксели
        valid_r = (point[1] + row_grid)[mask].astype(int32)
        valid_c = (point[0] + col_grid)[mask].astype(int32)
        valid_row_bin = row_bin[mask]
        valid_col_bin = col_bin[mask]

        # Градиенты
        dx = img[valid_r, valid_c + 1] - img[valid_r, valid_c - 1]
        dy = img[valid_r - 1, valid_c] - img[valid_r + 1, valid_c]

        magnitude = sqrt(dx * dx + dy * dy)
        orientation = (rad2deg(arctan2(dy, dx)) - angle) % 360.0

        # Веса
        weight = exp(-2.0 * ((valid_row_bin / hist_width) ** 2 +
                             (valid_col_bin / hist_width) ** 2) / 16.0)

        weighted_mag = weight * magnitude
        orientation_bin = orientation * (8.0 / 360.0)  # 8/360 = bins_per_degree

        # Гистограмма
        hist = zeros((6, 6, 8))

        for idx in range(len(valid_row_bin)):
            r_bin, c_bin = valid_row_bin[idx], valid_col_bin[idx]
            mag = weighted_mag[idx]
            o_bin = orientation_bin[idx]

            r_floor, c_floor, o_floor = floor([r_bin, c_bin, o_bin]).astype(int32)
            r_frac, c_frac, o_frac = r_bin - r_floor, c_bin - c_floor, o_bin - o_floor

            # Корректируем бины
            if o_floor < 0:
                o_floor += 8
            elif o_floor >= 8:
                o_floor -= 8

            # Трилинейная интерполяция
            c0 = mag * (1 - r_frac)
            c1 = mag * r_frac

            c00 = c0 * (1 - c_frac)
            c01 = c0 * c_frac
            c10 = c1 * (1 - c_frac)
            c11 = c1 * c_frac

            c000 = c00 * (1 - o_frac)
            c001 = c00 * o_frac
            c010 = c01 * (1 - o_frac)
            c011 = c01 * o_frac
            c100 = c10 * (1 - o_frac)
            c101 = c10 * o_frac
            c110 = c11 * (1 - o_frac)
            c111 = c11 * o_frac

            # Распределяем
            r_idx, c_idx = r_floor + 1, c_floor + 1
            o_idx1, o_idx2 = o_floor, (o_floor + 1) % 8

            hist[r_idx, c_idx, o_idx1] += c000
            hist[r_idx, c_idx, o_idx2] += c001
            hist[r_idx, c_idx + 1, o_idx1] += c010
            hist[r_idx, c_idx + 1, o_idx2] += c011
            hist[r_idx + 1, c_idx, o_idx1] += c100
            hist[r_idx + 1, c_idx, o_idx2] += c101
            hist[r_idx + 1, c_idx + 1, o_idx1] += c110
            hist[r_idx + 1, c_idx + 1, o_idx2] += c111

        # Извлекаем и нормализуем
        descriptor = hist[1:5, 1:5, :].flatten()
        norm_val = norm(descriptor)

        if norm_val > float_tolerance:
            threshold = norm_val * 0.2
            descriptor[descriptor > threshold] = threshold
            descriptor /= norm(descriptor)

        # Конвертация
        descriptor = (descriptor * 512).round()
        descriptor.clip(0, 255, out=descriptor)

        descriptors.append(descriptor.astype('float32'))

    return array(descriptors, dtype='float32')


# Вспомогательные функции
def ones(shape, dtype=float32):
    """Упрощенная функция ones"""
    if isinstance(shape, int):
        return array([1.0] * shape, dtype=dtype)
    else:
        # Для простоты - только для 1D
        return array([1.0] * shape[0], dtype=dtype)