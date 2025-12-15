"""Заглушка для utils4e для тестов"""

import numpy as np

def gaussian_kernel_2D(size=3, sigma=1.0):
    """Простой гауссовский фильтр"""
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) *
                     np.exp(-((x - (size-1)/2)**2 + (y - (size-1)/2)**2) / (2*sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)