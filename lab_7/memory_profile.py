from memory_profiler import profile
import numpy as np
from pysift import computeKeypointsAndDescriptors


@profile
def run_sift():
    # Используем меньшее изображение для профилирования памяти
    image = np.random.rand(128, 128) * 255
    image = image.astype(np.uint8)

    print(f"Image shape: {image.shape}")

    keypoints, descriptors = computeKeypointsAndDescriptors(image)
    print(f"Found {len(keypoints)} keypoints")
    print(f"Descriptors shape: {descriptors.shape if descriptors is not None else 'None'}")

    return keypoints, descriptors


if __name__ == '__main__':
    keypoints, descriptors = run_sift()