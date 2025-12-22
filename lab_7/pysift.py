from numpy import all, any, array, arctan2, cos, sin, exp, dot, log, logical_and, roll, sqrt, stack, trace, \
    unravel_index, pi, deg2rad, rad2deg, where, zeros, floor, full, nan, isnan, round, float32, int32, uint8, meshgrid
from numpy.linalg import det, lstsq, norm
from cv2 import resize, GaussianBlur, subtract, KeyPoint, INTER_LINEAR, INTER_NEAREST
from functools import cmp_to_key
import logging

####################
# Global variables #
####################

logger = logging.getLogger(__name__)
float_tolerance = 1e-7

# Предварительно вычисленные константы для оптимизации
SQRT_2 = sqrt(2.0)
TWO_PI = 2.0 * pi
DEG_TO_BIN = 36 / 360.0
BIN_TO_DEG = 360.0 / 36
INV_255 = 1.0 / 255.0
HIST_SMOOTH_COEFFS = array([1, 4, 6, 4, 1]) / 16.0  # Коэффициенты сглаживания гистограммы


#################
# Main function #
#################

def computeKeypointsAndDescriptors(image, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5):
    """Compute SIFT keypoints and descriptors for an input image
    """
    image = image.astype('float32')
    base_image = generateBaseImage(image, sigma, assumed_blur)
    num_octaves = computeNumberOfOctaves(base_image.shape)
    gaussian_kernels = generateGaussianKernels(sigma, num_intervals)
    gaussian_images = generateGaussianImages(base_image, num_octaves, gaussian_kernels)
    dog_images = generateDoGImages(gaussian_images)
    keypoints = findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width)
    keypoints = removeDuplicateKeypoints(keypoints)
    keypoints = convertKeypointsToInputImageSize(keypoints)
    descriptors = generateDescriptorsOptimized(keypoints, gaussian_images)
    return keypoints, descriptors


#########################
# Image pyramid related #
#########################

def generateBaseImage(image, sigma, assumed_blur):
    """Generate base image from input image by upsampling by 2 in both directions and blurring
    """
    logger.debug('Generating base image...')
    image = resize(image, (0, 0), fx=2, fy=2, interpolation=INTER_LINEAR)
    assumed_blur_sq = (2 * assumed_blur) ** 2
    sigma_diff_sq = max((sigma ** 2) - assumed_blur_sq, 0.01)
    sigma_diff = sqrt(sigma_diff_sq)
    return GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)


def computeNumberOfOctaves(image_shape):
    """Compute number of octaves in image pyramid as function of base image shape (OpenCV default)
    """
    return int(round(log(min(image_shape)) / log(2) - 1))


def generateGaussianKernels(sigma, num_intervals):
    """Generate list of gaussian kernels at which to blur the input image.
    """
    logger.debug('Generating scales...')
    num_images_per_octave = num_intervals + 3
    k = 2 ** (1. / num_intervals)
    gaussian_kernels = zeros(num_images_per_octave)
    gaussian_kernels[0] = sigma

    k_powers = [k ** i for i in range(num_images_per_octave)]

    for image_index in range(1, num_images_per_octave):
        sigma_previous = k_powers[image_index - 1] * sigma
        sigma_total = k * sigma_previous
        gaussian_kernels[image_index] = sqrt(sigma_total ** 2 - sigma_previous ** 2)
    return gaussian_kernels


def generateGaussianImages(image, num_octaves, gaussian_kernels):
    """Generate scale-space pyramid of Gaussian images
    """
    logger.debug('Generating Gaussian images...')
    gaussian_images = []

    current_image = image

    for octave_index in range(num_octaves):
        gaussian_images_in_octave = []
        gaussian_images_in_octave.append(current_image)

        kernels_to_apply = gaussian_kernels[1:]
        for gaussian_kernel in kernels_to_apply:
            current_image = GaussianBlur(current_image, (0, 0), sigmaX=gaussian_kernel, sigmaY=gaussian_kernel)
            gaussian_images_in_octave.append(current_image)

        gaussian_images.append(gaussian_images_in_octave)

        if octave_index < num_octaves - 1:
            octave_base = gaussian_images_in_octave[-3]
            h, w = octave_base.shape[:2]
            current_image = resize(octave_base, (w // 2, h // 2), interpolation=INTER_NEAREST)

    return array(gaussian_images, dtype=object)


def generateDoGImages(gaussian_images):
    """Generate Difference-of-Gaussians image pyramid
    """
    logger.debug('Generating Difference-of-Gaussian images...')
    dog_images = []

    for gaussian_images_in_octave in gaussian_images:
        dog_images_in_octave = []
        num_images = len(gaussian_images_in_octave)
        for i in range(num_images - 1):
            dog_images_in_octave.append(subtract(gaussian_images_in_octave[i + 1], gaussian_images_in_octave[i]))
        dog_images.append(dog_images_in_octave)

    return array(dog_images, dtype=object)


###############################
# Scale-space extrema related #
###############################

def findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width,
                          contrast_threshold=0.04):
    """Find pixel positions of all scale-space extrema in the image pyramid
    """
    logger.debug('Finding scale-space extrema...')
    threshold_val = floor(0.5 * contrast_threshold / num_intervals * 255)
    keypoints = []

    for octave_index, dog_images_in_octave in enumerate(dog_images):
        first_img_shape = dog_images_in_octave[0].shape

        row_start = image_border_width
        row_end = first_img_shape[0] - image_border_width
        col_start = image_border_width
        col_end = first_img_shape[1] - image_border_width

        num_dog_images = len(dog_images_in_octave)

        for image_index in range(1, num_dog_images - 1):
            prev_img = dog_images_in_octave[image_index - 1]
            curr_img = dog_images_in_octave[image_index]
            next_img = dog_images_in_octave[image_index + 1]

            for i in range(row_start, row_end):
                prev_row = prev_img[i - 1:i + 2, col_start - 1:col_end + 1]
                curr_row = curr_img[i - 1:i + 2, col_start - 1:col_end + 1]
                next_row = next_img[i - 1:i + 2, col_start - 1:col_end + 1]

                center_pixels = curr_img[i, col_start:col_end]

                for j_offset, center_val in enumerate(center_pixels):
                    if abs(center_val) > threshold_val:
                        j = col_start + j_offset

                        if isPixelAnExtremumOptimized(
                                prev_row[:, j_offset:j_offset + 3],
                                curr_row[:, j_offset:j_offset + 3],
                                next_row[:, j_offset:j_offset + 3],
                                center_val
                        ):
                            localization_result = localizeExtremumViaQuadraticFitOptimized(
                                i, j, image_index, octave_index, num_intervals,
                                dog_images_in_octave, sigma, contrast_threshold,
                                image_border_width, threshold_val
                            )
                            if localization_result is not None:
                                keypoint, localized_image_index = localization_result
                                keypoints_with_orientations = computeKeypointsWithOrientationsOptimized(
                                    keypoint, octave_index, gaussian_images[octave_index][localized_image_index]
                                )
                                if keypoints_with_orientations:
                                    keypoints.extend(keypoints_with_orientations)

    return keypoints


def isPixelAnExtremumOptimized(prev_slice, curr_slice, next_slice, center_value):
    """Highly optimized extremum check with early exits"""
    if center_value > 0:
        if (center_value < curr_slice[0, 0] or center_value < curr_slice[0, 1] or center_value < curr_slice[0, 2] or
                center_value < curr_slice[1, 0] or center_value < curr_slice[1, 2] or
                center_value < curr_slice[2, 0] or center_value < curr_slice[2, 1] or center_value < curr_slice[2, 2]):
            return False

        if (center_value < prev_slice[0, 0] or center_value < prev_slice[0, 1] or center_value < prev_slice[0, 2] or
                center_value < prev_slice[1, 0] or center_value < prev_slice[1, 1] or center_value < prev_slice[1, 2] or
                center_value < prev_slice[2, 0] or center_value < prev_slice[2, 1] or center_value < prev_slice[2, 2]):
            return False

        if (center_value < next_slice[0, 0] or center_value < next_slice[0, 1] or center_value < next_slice[0, 2] or
                center_value < next_slice[1, 0] or center_value < next_slice[1, 1] or center_value < next_slice[1, 2] or
                center_value < next_slice[2, 0] or center_value < next_slice[2, 1] or center_value < next_slice[2, 2]):
            return False

        return True
    elif center_value < 0:
        if (center_value > curr_slice[0, 0] or center_value > curr_slice[0, 1] or center_value > curr_slice[0, 2] or
                center_value > curr_slice[1, 0] or center_value > curr_slice[1, 2] or
                center_value > curr_slice[2, 0] or center_value > curr_slice[2, 1] or center_value > curr_slice[2, 2]):
            return False

        if (center_value > prev_slice[0, 0] or center_value > prev_slice[0, 1] or center_value > prev_slice[0, 2] or
                center_value > prev_slice[1, 0] or center_value > prev_slice[1, 1] or center_value > prev_slice[1, 2] or
                center_value > prev_slice[2, 0] or center_value > prev_slice[2, 1] or center_value > prev_slice[2, 2]):
            return False

        if (center_value > next_slice[0, 0] or center_value > next_slice[0, 1] or center_value > next_slice[0, 2] or
                center_value > next_slice[1, 0] or center_value > next_slice[1, 1] or center_value > next_slice[1, 2] or
                center_value > next_slice[2, 0] or center_value > next_slice[2, 1] or center_value > next_slice[2, 2]):
            return False

        return True
    return False


def localizeExtremumViaQuadraticFitOptimized(i, j, image_index, octave_index, num_intervals, dog_images_in_octave,
                                             sigma, contrast_threshold, image_border_width, initial_threshold,
                                             eigenvalue_ratio=10, num_attempts_until_convergence=5):
    """Optimized version of extremum localization"""
    logger.debug('Localizing scale-space extrema (optimized)...')

    image_shape = dog_images_in_octave[0].shape
    height, width = image_shape

    contrast_threshold_scaled = contrast_threshold / num_intervals
    octave_scale_factor = 2 ** octave_index

    for attempt_index in range(num_attempts_until_convergence):
        img_idx = image_index
        first_image = dog_images_in_octave[img_idx - 1]
        second_image = dog_images_in_octave[img_idx]
        third_image = dog_images_in_octave[img_idx + 1]

        i_minus_1, i_plus_2 = i - 1, i + 2
        j_minus_1, j_plus_2 = j - 1, j + 2

        block1 = first_image[i_minus_1:i_plus_2, j_minus_1:j_plus_2].astype('float32')
        block2 = second_image[i_minus_1:i_plus_2, j_minus_1:j_plus_2].astype('float32')
        block3 = third_image[i_minus_1:i_plus_2, j_minus_1:j_plus_2].astype('float32')

        pixel_cube = stack([block1, block2, block3]) * INV_255

        gradient = computeGradientAtCenterPixelOptimized(pixel_cube)
        hessian = computeHessianAtCenterPixelOptimized(pixel_cube)

        try:
            extremum_update = -lstsq(hessian, gradient, rcond=None)[0]
        except:
            return None

        if (abs(extremum_update[0]) < 0.5 and
                abs(extremum_update[1]) < 0.5 and
                abs(extremum_update[2]) < 0.5):
            break

        j_new = j + int(round(extremum_update[0]))
        i_new = i + int(round(extremum_update[1]))
        image_index_new = img_idx + int(round(extremum_update[2]))

        if (i_new < image_border_width or i_new >= height - image_border_width or
                j_new < image_border_width or j_new >= width - image_border_width or
                image_index_new < 1 or image_index_new > num_intervals):
            return None

        i, j, image_index = i_new, j_new, image_index_new

    if attempt_index >= num_attempts_until_convergence - 1:
        return None

    center_value = pixel_cube[1, 1, 1]
    functionValueAtUpdatedExtremum = center_value + 0.5 * dot(gradient, extremum_update)

    if abs(functionValueAtUpdatedExtremum) < contrast_threshold_scaled:
        return None

    xy_hessian = hessian[:2, :2]
    xy_hessian_trace = trace(xy_hessian)
    xy_hessian_det = det(xy_hessian)

    if xy_hessian_det <= 0:
        return None

    trace_sq = xy_hessian_trace ** 2
    if eigenvalue_ratio * trace_sq >= ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
        return None

    keypoint = KeyPoint()

    j_final = (j + extremum_update[0]) * octave_scale_factor
    i_final = (i + extremum_update[1]) * octave_scale_factor
    keypoint.pt = (j_final, i_final)

    scale_offset = int(round((extremum_update[2] + 0.5) * 255))
    keypoint.octave = (octave_index & 255) | ((image_index & 255) << 8) | ((scale_offset & 255) << 16)

    scale_exp = (image_index + extremum_update[2]) / float32(num_intervals)
    keypoint.size = sigma * (2 ** scale_exp) * (2 ** (octave_index + 1))

    keypoint.response = abs(functionValueAtUpdatedExtremum)

    return keypoint, image_index


def computeGradientAtCenterPixelOptimized(pixel_array):
    """Optimized gradient computation"""
    dx = (pixel_array[1, 1, 2] - pixel_array[1, 1, 0]) * 0.5
    dy = (pixel_array[1, 2, 1] - pixel_array[1, 0, 1]) * 0.5
    ds = (pixel_array[2, 1, 1] - pixel_array[0, 1, 1]) * 0.5

    return array([dx, dy, ds], dtype='float32')


def computeHessianAtCenterPixelOptimized(pixel_array):
    """Optimized Hessian computation"""
    center = pixel_array[1, 1, 1]

    dxx = pixel_array[1, 1, 2] - 2.0 * center + pixel_array[1, 1, 0]
    dyy = pixel_array[1, 2, 1] - 2.0 * center + pixel_array[1, 0, 1]
    dss = pixel_array[2, 1, 1] - 2.0 * center + pixel_array[0, 1, 1]

    dxy = (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] -
           pixel_array[1, 0, 2] + pixel_array[1, 0, 0]) * 0.25

    dxs = (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] -
           pixel_array[0, 1, 2] + pixel_array[0, 1, 0]) * 0.25

    dys = (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] -
           pixel_array[0, 2, 1] + pixel_array[0, 0, 1]) * 0.25

    return array([[dxx, dxy, dxs],
                  [dxy, dyy, dys],
                  [dxs, dys, dss]], dtype='float32')


#########################
# Keypoint orientations #
#########################

def computeKeypointsWithOrientationsOptimized(keypoint, octave_index, gaussian_image, radius_factor=3, num_bins=36,
                                              peak_ratio=0.8, scale_factor=1.5):
    """Optimized version for computing keypoint orientations"""
    logger.debug('Computing keypoint orientations (optimized)...')
    keypoints_with_orientations = []
    image_shape = gaussian_image.shape

    # Предварительные вычисления
    scale = scale_factor * keypoint.size / float32(2 ** (octave_index + 1))
    radius = int(round(radius_factor * scale))
    weight_factor = -0.5 / (scale ** 2)

    base_y = int(round(keypoint.pt[1] / float32(2 ** octave_index)))
    base_x = int(round(keypoint.pt[0] / float32(2 ** octave_index)))

    # Определяем границы региона
    y_min = max(1, base_y - radius)
    y_max = min(image_shape[0] - 2, base_y + radius)
    x_min = max(1, base_x - radius)
    x_max = min(image_shape[1] - 2, base_x + radius)

    if y_min >= y_max or x_min >= x_max:
        return keypoints_with_orientations

    # Вычисляем градиенты для всего региона сразу
    region_height = y_max - y_min + 1
    region_width = x_max - x_min + 1

    # Получаем регион изображения
    region = gaussian_image[y_min - 1:y_max + 2, x_min - 1:x_max + 2]

    # Вычисляем градиенты через срезы
    dx = region[1:region_height + 1, 2:region_width + 2] - region[1:region_height + 1, 0:region_width]
    dy = region[0:region_height, 1:region_width + 1] - region[2:region_height + 2, 1:region_width + 1]

    # Вычисляем магнитуду и ориентацию
    gradient_magnitude = sqrt(dx * dx + dy * dy)
    gradient_orientation = rad2deg(arctan2(dy, dx))

    # Создаем гистограмму
    raw_histogram = zeros(num_bins)

    # Заполняем гистограмму
    for i in range(region_height):
        for j in range(region_width):
            y_offset = (y_min + i) - base_y
            x_offset = (x_min + j) - base_x

            weight = exp(weight_factor * (y_offset ** 2 + x_offset ** 2))
            histogram_index = int(round(gradient_orientation[i, j] * DEG_TO_BIN)) % num_bins
            raw_histogram[histogram_index] += weight * gradient_magnitude[i, j]

    # Быстрое сглаживание гистограммы
    smooth_histogram = zeros(num_bins)
    for n in range(num_bins):
        smooth_histogram[n] = (
                HIST_SMOOTH_COEFFS[0] * raw_histogram[(n - 2) % num_bins] +
                HIST_SMOOTH_COEFFS[1] * raw_histogram[(n - 1) % num_bins] +
                HIST_SMOOTH_COEFFS[2] * raw_histogram[n] +
                HIST_SMOOTH_COEFFS[3] * raw_histogram[(n + 1) % num_bins] +
                HIST_SMOOTH_COEFFS[4] * raw_histogram[(n + 2) % num_bins]
        )

    # Находим пики
    orientation_max = smooth_histogram.max()
    if orientation_max < float_tolerance:
        return keypoints_with_orientations

    shifted_left = roll(smooth_histogram, 1)
    shifted_right = roll(smooth_histogram, -1)
    orientation_peaks = where(logical_and(smooth_histogram > shifted_left,
                                          smooth_histogram > shifted_right))[0]

    peak_threshold = peak_ratio * orientation_max

    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]
        if peak_value >= peak_threshold:
            # Интерполяция пика
            left_value = smooth_histogram[(peak_index - 1) % num_bins]
            right_value = smooth_histogram[(peak_index + 1) % num_bins]

            interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) /
                                       (left_value - 2 * peak_value + right_value)) % num_bins

            orientation = 360. - interpolated_peak_index * BIN_TO_DEG
            if abs(orientation - 360.) < float_tolerance:
                orientation = 0

            new_keypoint = KeyPoint(*keypoint.pt, keypoint.size, orientation,
                                    keypoint.response, keypoint.octave)
            keypoints_with_orientations.append(new_keypoint)

    return keypoints_with_orientations


##############################
# Duplicate keypoint removal #
##############################

def compareKeypoints(keypoint1, keypoint2):
    """Return True if keypoint1 is less than keypoint2
    """
    if keypoint1.pt[0] != keypoint2.pt[0]:
        return keypoint1.pt[0] - keypoint2.pt[0]
    if keypoint1.pt[1] != keypoint2.pt[1]:
        return keypoint1.pt[1] - keypoint2.pt[1]
    if keypoint1.size != keypoint2.size:
        return keypoint2.size - keypoint1.size
    if keypoint1.angle != keypoint2.angle:
        return keypoint1.angle - keypoint2.angle
    if keypoint1.response != keypoint2.response:
        return keypoint2.response - keypoint1.response
    if keypoint1.octave != keypoint2.octave:
        return keypoint2.octave - keypoint1.octave
    return keypoint2.class_id - keypoint1.class_id


def removeDuplicateKeypoints(keypoints):
    """Sort keypoints and remove duplicate keypoints
    """
    if len(keypoints) < 2:
        return keypoints

    keypoints.sort(key=cmp_to_key(compareKeypoints))
    unique_keypoints = [keypoints[0]]

    for next_keypoint in keypoints[1:]:
        last_unique_keypoint = unique_keypoints[-1]
        if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or \
                last_unique_keypoint.pt[1] != next_keypoint.pt[1] or \
                last_unique_keypoint.size != next_keypoint.size or \
                last_unique_keypoint.angle != next_keypoint.angle:
            unique_keypoints.append(next_keypoint)
    return unique_keypoints


#############################
# Keypoint scale conversion #
#############################

def convertKeypointsToInputImageSize(keypoints):
    """Convert keypoint point, size, and octave to input image size
    """
    converted_keypoints = []
    for keypoint in keypoints:
        keypoint.pt = tuple(0.5 * array(keypoint.pt))
        keypoint.size *= 0.5
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
        converted_keypoints.append(keypoint)
    return converted_keypoints


#########################
# Descriptor generation #
#########################

def unpackOctave(keypoint):
    """Compute octave, layer, and scale from a keypoint
    """
    octave = keypoint.octave & 255
    layer = (keypoint.octave >> 8) & 255
    if octave >= 128:
        octave = octave | -128
    scale = 1 / float32(1 << octave) if octave >= 0 else float32(1 << -octave)
    return octave, layer, scale


def generateDescriptorsOptimized(keypoints, gaussian_images, window_width=4, num_bins=8, scale_multiplier=3,
                                 descriptor_max_value=0.2):
    """Optimized descriptor generation"""
    logger.debug('Generating descriptors (optimized)...')
    descriptors = []

    # Предварительно вычисленные константы
    half_window = window_width * 0.5
    weight_multiplier = -2.0 / (window_width ** 2)  # -0.5 / ((0.5 * window_width) ** 2)
    bins_per_degree = num_bins / 360.0

    for keypoint in keypoints:
        octave, layer, scale = unpackOctave(keypoint)
        gaussian_image = gaussian_images[octave + 1, layer]
        num_rows, num_cols = gaussian_image.shape

        point = (scale * array(keypoint.pt)).round().astype(int32)
        angle = 360.0 - keypoint.angle
        angle_rad = deg2rad(angle)
        cos_a = cos(angle_rad)
        sin_a = sin(angle_rad)

        # Размер окна дескриптора
        hist_width = scale_multiplier * 0.5 * scale * keypoint.size
        half_width = int(min(
            round(hist_width * SQRT_2 * (window_width + 1) * 0.5),
            sqrt(num_rows ** 2 + num_cols ** 2)
        ))

        # Определяем границы региона
        row_min = max(1, point[1] - half_width)
        row_max = min(num_rows - 2, point[1] + half_width)
        col_min = max(1, point[0] - half_width)
        col_max = min(num_cols - 2, point[0] + half_width)

        if row_min >= row_max or col_min >= col_max:
            descriptors.append(zeros(window_width * window_width * num_bins, dtype='float32'))
            continue

        # Создаем сетку координат
        rows = arange(row_min, row_max + 1) - point[1]
        cols = arange(col_min, col_max + 1) - point[0]
        col_grid, row_grid = meshgrid(cols, rows)

        # Вращаем координаты
        row_rot = col_grid * sin_a + row_grid * cos_a
        col_rot = col_grid * cos_a - row_grid * sin_a

        # Преобразуем в бины
        row_bin = (row_rot / hist_width) + half_window - 0.5
        col_bin = (col_rot / hist_width) + half_window - 0.5

        # Маска валидных бинов
        valid_mask = (row_bin > -1) & (row_bin < window_width) & \
                     (col_bin > -1) & (col_bin < window_width)

        if not valid_mask.any():
            descriptors.append(zeros(window_width * window_width * num_bins, dtype='float32'))
            continue

        # Получаем индексы пикселей
        window_rows = point[1] + row_grid
        window_cols = point[0] + col_grid

        # Вычисляем градиенты для валидных пикселей
        valid_rows = window_rows[valid_mask].astype(int32)
        valid_cols = window_cols[valid_mask].astype(int32)
        valid_row_bin = row_bin[valid_mask]
        valid_col_bin = col_bin[valid_mask]

        # Векторизованное вычисление градиентов
        dx = gaussian_image[valid_rows, valid_cols + 1] - gaussian_image[valid_rows, valid_cols - 1]
        dy = gaussian_image[valid_rows - 1, valid_cols] - gaussian_image[valid_rows + 1, valid_cols]

        gradient_magnitude = sqrt(dx * dx + dy * dy)
        gradient_orientation = (rad2deg(arctan2(dy, dx)) - angle) % 360.0

        # Веса
        weight = exp(weight_multiplier * ((valid_row_bin / hist_width) ** 2 +
                                          (valid_col_bin / hist_width) ** 2))

        weighted_magnitude = weight * gradient_magnitude
        orientation_bin = gradient_orientation * bins_per_degree

        # Создаем гистограмму
        histogram_tensor = zeros((window_width + 2, window_width + 2, num_bins))

        # Быстрое распределение по бинам
        for idx in range(len(valid_row_bin)):
            r_bin, c_bin = valid_row_bin[idx], valid_col_bin[idx]
            mag = weighted_magnitude[idx]
            o_bin = orientation_bin[idx]

            r_floor, c_floor, o_floor = floor([r_bin, c_bin, o_bin]).astype(int32)
            r_frac, c_frac, o_frac = r_bin - r_floor, c_bin - c_floor, o_bin - o_floor

            # Корректируем бины ориентации
            if o_floor < 0:
                o_floor += num_bins
            elif o_floor >= num_bins:
                o_floor -= num_bins

            # Вычисляем веса для 8 соседних бинов
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

            # Распределяем по бинам
            r_idx, c_idx = r_floor + 1, c_floor + 1
            o_idx1 = o_floor
            o_idx2 = (o_floor + 1) % num_bins

            histogram_tensor[r_idx, c_idx, o_idx1] += c000
            histogram_tensor[r_idx, c_idx, o_idx2] += c001
            histogram_tensor[r_idx, c_idx + 1, o_idx1] += c010
            histogram_tensor[r_idx, c_idx + 1, o_idx2] += c011
            histogram_tensor[r_idx + 1, c_idx, o_idx1] += c100
            histogram_tensor[r_idx + 1, c_idx, o_idx2] += c101
            histogram_tensor[r_idx + 1, c_idx + 1, o_idx1] += c110
            histogram_tensor[r_idx + 1, c_idx + 1, o_idx2] += c111

        # Извлекаем и нормализуем дескриптор
        descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()

        norm_val = norm(descriptor_vector)
        if norm_val > float_tolerance:
            threshold = norm_val * descriptor_max_value
            descriptor_vector[descriptor_vector > threshold] = threshold
            descriptor_vector /= norm(descriptor_vector)

        # Конвертация
        descriptor_vector = (descriptor_vector * 512).round()
        descriptor_vector.clip(0, 255, out=descriptor_vector)

        descriptors.append(descriptor_vector.astype('float32'))

    return array(descriptors, dtype='float32')


# Вспомогательные функции
def arange(start, stop=None, step=1, dtype=None):
    """Упрощенная версия arange"""
    if stop is None:
        stop = start
        start = 0
    return array([start + i * step for i in range(int((stop - start) / step))], dtype=dtype)