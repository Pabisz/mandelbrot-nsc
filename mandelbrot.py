"""
Mandelbrot Set Generator

Author: [Sebastian Pabisz Frolund]
Course: Numerical Scientific Computing 2026
"""

import numpy as np
import matplotlib.pyplot as plt
#import time


def mandelbrot_point(c, max_iter):
    """
    Determines the number of iterations for a point in the Mandelbrot set.

    Parameters:
    c (complex): The complex point to evaluate.
    max_iter (int): The maximum number of iterations to perform.

    Returns:
    int: The number of iterations before divergence, or max_iter if it does not diverge.
    """
    z = 0
    for n in range(max_iter):
        z = z * z + c
        if abs(z) > 2:
            return n
    return max_iter

def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    """
    Generates the Mandelbrot set for a given region and resolution.

    Parameters:
    xmin, xmax (float): The range of the real part of the complex plane.
    ymin, ymax (float): The range of the imaginary part of the complex plane.
    width, height (int): The resolution of the output image.
    max_iter (int): The maximum number of iterations to perform.

    Returns:
    np.ndarray: A 2D array representing the Mandelbrot set.
    """
    result = np.zeros((height, width), dtype=int)

    x_values = np.linspace(xmin, xmax, width)
    y_values = np.linspace(ymin, ymax, height)

    for i in range(height):
        for j in range(width):
            c = complex(x_values[j], y_values[i])
            result[i, j] = mandelbrot_point(c, max_iter)

    return result

print(mandelbrot_point(complex(0, 0), 100))
