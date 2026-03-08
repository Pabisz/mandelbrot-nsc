"""
Parallelized Mandelbrot Set Generator

Author: [Sebastian Pabisz Frolund]
Course: Numerical Scientific Computing 2026
"""
import numpy as np
from numba import njit
from multiprocessing import Pool
import time, os, statistics, matplotlib.pyplot as plt
from pathlib import Path

@njit
def mandelbrot_pixel(c_real, c_imag, max_iter =100) :
    z_real = z_imag = 0.0
    for n in range ( max_iter ) :
        zr2 = z_real * z_real
        zi2 = z_imag * z_imag
        if zr2 + zi2 > 4.0: return n
        z_imag, z_real = 2.0 * z_real * z_imag + c_imag, zr2 - zi2 + c_real
    return max_iter

@njit
def mandelbrot_chunk(rowstart, rowend, N, x_min, x_max, y_min, y_max, max_iter):
    output = np.empty((rowend - rowstart, N), dtype=np.int32)
    dx = (x_max - x_min) / (N - 1)
    dy = (y_max - y_min) / (N - 1)
    for i in range(rowend-rowstart):
        c_imag = y_min + (rowstart + i) * dy
        for col in range(N):
            output[i, col] = mandelbrot_pixel(x_min + col * dx, c_imag, max_iter)
    return output

def mandelbrot_serial(x_min, x_max, y_min, y_max, N, max_iter):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)

if __name__ == "__main__":

    N, MAX_ITER = 1024, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2, 1, -1.5, 1.5

    result = mandelbrot_serial(X_MIN, X_MAX,Y_MIN, Y_MAX,N, MAX_ITER)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(result, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], cmap="viridis",origin="lower",aspect="equal")

    ax.set_xlabel("Re(c)")
    ax.set_ylabel("Im(c)")
    ax.set_title("Parallel Mandelbrot")

    plt.show()