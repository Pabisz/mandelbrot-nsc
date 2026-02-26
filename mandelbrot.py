"""
Mandelbrot Set Generator

Author: [Sebastian Pabisz Frolund]
Course: Numerical Scientific Computing 2026
"""
import numpy as np
import matplotlib.pyplot as plt
import time, statistics


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
        if abs(z) > 2:
            return n
        z = z * z + c
    return max_iter

def mandelbrot_set_naive(xmin, xmax, ymin, ymax, width, height, max_iter):
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

def mandelbrot_set_vectorized(xmin, xmax, ymin, ymax, width, height, max_iter):
    """
    Generates the Mandelbrot set using vectorized operations for improved performance.

    Parameters:
    xmin, xmax (float): The range of the real part of the complex plane.
    ymin, ymax (float): The range of the imaginary part of the complex plane.
    width, height (int): The resolution of the output image.
    max_iter (int): The maximum number of iterations to perform.

    Returns:
    np.ndarray: A 2D array representing the Mandelbrot set.
    """
    x_values = np.linspace(xmin, xmax, width)
    y_values = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x_values, y_values)
    C = X + 1j * Y
    
    Z = np.zeros_like(C, dtype=complex)
    result = np.zeros(C.shape, dtype=int)
    for n in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask] * Z[mask] + C[mask]
        result[mask] += 1
    return result

def bench (fn , * args , runs =5) :
    fn (* args ) # warm -up
    times = []
    for _ in range ( runs ) :
        t0 = time . perf_counter ()
        fn (* args )
        times . append ( time . perf_counter () - t0 )
    return statistics . median ( times )

if __name__ == "__main__":
    # Parameters for the Mandelbrot set
    xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
    width, height = 1024, 1024
    max_iter = 100
    args = (xmin, xmax, ymin, ymax, width, height, 100)

    t_naive = bench(mandelbrot_set_naive, *args)
    t_vectorized = bench(mandelbrot_set_vectorized, *args)
    print(f"Naive implementation took {t_naive:.4f} seconds")
    print(f"Vectorized implementation took {t_vectorized:.4f} seconds")

    # Plot the Mandelbrot set using vectorized implementation
    mandelbrot_image = mandelbrot_set_vectorized(*args)
    plt.imshow(mandelbrot_image, extent=(xmin, xmax, ymin, ymax),cmap='viridis')
    plt.colorbar()
    plt.title('Mandelbrot Set')
    plt.xlabel('Real Axis')
    plt.ylabel('Imaginary Axis')
    plt.show()
