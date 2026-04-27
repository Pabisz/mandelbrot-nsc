"""
Parallelized Mandelbrot Set Generator

Author: [Sebastian Pabisz Frolund]
Course: Numerical Scientific Computing 2026
"""

import multiprocessing

import numpy as np
from numba import njit
from multiprocessing import Pool
import time
import os
import statistics
import matplotlib.pyplot as plt
from dask.distributed import Client, LocalCluster
import dask


@njit(cache=True)
def mandelbrot_pixel(c_real: float, c_imag: float, max_iter: int = 100) -> int:
    """Calculate the Mandelbrot iteration count for a given complex number.

    Args:
        c_real (float): Real part of the complex number.
        c_imag (float): Imaginary part of the complex number.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.

    Returns:
        int: Number of iterations before divergence, or max_iter if it does not diverge.
    """
    z_real = z_imag = 0.0
    for n in range(max_iter):
        zr2 = z_real * z_real
        zi2 = z_imag * z_imag
        if zr2 + zi2 > 4.0:
            return n
        z_imag, z_real = 2.0 * z_real * z_imag + c_imag, zr2 - zi2 + c_real
    return max_iter


@njit(cache=True)
def mandelbrot_chunk(
    rowstart: int,
    rowend: int,
    N: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    max_iter: int,
) -> np.ndarray:
    """Computes a horizontal chunk of the Mandelbrot set.

    Args:
        rowstart (int): Starting row index (inclusive) in the full image grid.
        rowend (int): Ending row index (exclusive) in the full image grid.
        N (int): Number of pixels in both x and y directions (image is NxN).
        x_min (float): Minimum real value of the complex plane.
        x_max (float): Maximum real value of the complex plane.
        y_min (float): Minimum imaginary value of the complex plane.
        y_max (float): Maximum imaginary value of the complex plane.
        max_iter (int): Maximum number of iterations for divergence test.

    Returns:
        np.ndarray: 2D array of shape (rowend - rowstart, N) containing
        iteration counts for each pixel in the specified chunk.
    """
    output = np.empty((rowend - rowstart, N), dtype=np.int32)
    dx = (x_max - x_min) / (N - 1)
    dy = (y_max - y_min) / (N - 1)
    for i in range(rowend - rowstart):
        c_imag = y_min + (rowstart + i) * dy
        for col in range(N):
            output[i, col] = mandelbrot_pixel(x_min + col * dx, c_imag, max_iter)
    return output


def mandelbrot_serial(
    x_min: float, x_max: float, y_min: float, y_max: float, N: int, max_iter: int
) -> np.ndarray:
    """Computes mandelbrot set for a given area (Runs mandelbrot_chunk with one chunk)

    Args:
        x_min (float): Minimum real value of the complex plane.
        x_max (float): Maximum real value of the complex plane.
        y_min (float): Minimum imaginary value of the complex plane.
        y_max (float): Maximum imaginary value of the complex plane.
        N (int): Number of pixels in both x and y directions (image is NxN).
        max_iter (int): Maximum number of iterations for divergence test.

    Returns:
        np.ndarray: 2D array of iteration counts for the specified area.
    """
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)


def _worker(args: tuple) -> np.ndarray:
    """Worker function for parallel Mandelbrot computation.

    Args:
        args (tuple): Tuple containing the arguments:
            (rowstart, rowend, N, x_min, x_max, y_min, y_max, max_iter)

    Returns:
        np.ndarray: 2D array containing iteration counts for the given chunk.
    """
    return mandelbrot_chunk(*args)


def mandelbrot_parallel(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    N: int,
    max_iter: int,
    n_workers: int = 4,
    n_chunks: int = None,
    pool: multiprocessing.Pool = None,
) -> np.ndarray:
    """Compute the Mandelbrot set using multiprocessing.

    Args:
        x_min (float): Minimum real value of the complex plane.
        x_max (float): Maximum real value of the complex plane.
        y_min (float): Minimum imaginary value of the complex plane.
        y_max (float): Maximum imaginary value of the complex plane.
        N (int): Number of pixels in both x and y directions (image is NxN).
        max_iter (int): Maximum number of iterations for divergence test.
        n_workers (int, optional): Number of worker processes. Defaults to 4.
        n_chunks (int, optional): Number of chunks to split the work into.
            Defaults to ``n_workers`` if not specified.
        pool (multiprocessing.Pool, optional): Existing pool to reuse.
            If None, a new pool is created.

    Returns:
        np.ndarray: 2D array of iteration counts for the full image.
    """
    if n_chunks is None:
        n_chunks = n_workers

    chunk_size = max(1, N // n_chunks)
    chunks, row = [], 0
    while row < N:
        end = min(row + chunk_size, N)
        chunks.append((row, end, N, x_min, x_max, y_min, y_max, max_iter))
        row = end
    if pool is not None:
        return np.vstack(pool.map(_worker, chunks))
    tiny = [(0, 8, 8, x_min, x_max, y_min, y_max, max_iter)]
    with Pool(processes=n_workers) as pool:
        pool.map(_worker, tiny)  # warm-up: Numba JIT in all workers
        return np.vstack(pool.map(_worker, chunks))


def mandelbrot_dask(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    N: int,
    max_iter: int,
    n_chunks: int = 48,
) -> np.ndarray:
    """Compute the Mandelbrot set using Dask for parallel execution.

    Args:
        x_min (float): Minimum real value of the complex plane.
        x_max (float): Maximum real value of the complex plane.
        y_min (float): Minimum imaginary value of the complex plane.
        y_max (float): Maximum imaginary value of the complex plane.
        N (int): Number of pixels in both x and y directions (image is NxN).
        max_iter (int): Maximum number of iterations for divergence test.
        n_chunks (int, optional): Number of chunks/tasks to create.
            Defaults to 48.

    Returns:
        np.ndarray: 2D array of iteration counts for the full image.
    """

    chunk_size = max(1, N // n_chunks)
    tasks, row = [], 0
    while row < N:
        end = min(row + chunk_size, N)
        tasks.append(
            dask.delayed(mandelbrot_chunk)(
                row, end, N, x_min, x_max, y_min, y_max, max_iter
            )
        )
        row = end
    parts = dask.compute(*tasks)
    return np.vstack(parts)


if __name__ == "__main__":
    N, MAX_ITER = 4096, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2, 1, -1.5, 1.5

    result = mandelbrot_parallel(X_MIN, X_MAX, Y_MIN, Y_MAX, N, MAX_ITER)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(
        result,
        extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
        cmap="viridis",
        origin="lower",
        aspect="equal",
    )

    ax.set_xlabel("Re(c)")
    ax.set_ylabel("Im(c)")
    ax.set_title("Parallel Mandelbrot")

    plt.show()

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_serial(X_MIN, X_MAX, Y_MIN, Y_MAX, N, MAX_ITER)
        times.append(time.perf_counter() - t0)
        t_serial = statistics.median(times)

    workers_list = []
    efficiency_list = []
    speedup_list = []
    tiny = [(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10)]

    for n_workers in range(1, os.cpu_count() + 1):
        chunk_size = max(1, N // n_workers)
        chunks, row = [], 0
        while row < N:
            end = min(row + chunk_size, N)
            chunks.append((row, end, N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER))
            row = end
        with Pool(processes=n_workers) as pool:
            pool.map(_worker, tiny)  # warm-up: Numba JIT in all workers
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                np.vstack(pool.map(_worker, chunks))
                times.append(time.perf_counter() - t0)
        t_par = statistics.median(times)
        speedup = t_serial / t_par
        efficiency = speedup / n_workers * 100
        workers_list.append(n_workers)
        speedup_list.append(speedup)
        efficiency_list.append(efficiency)
        print(  
            f"{n_workers:2d} workers: {t_par:.3f}s, speedup={speedup:.2f}x, eff={efficiency:.0f}%"
        )

    plt.figure(figsize=(8, 6))
    plt.plot(workers_list, speedup_list, marker="o", label="Measured speedup")
    plt.plot(workers_list, workers_list, "--", label="Ideal speedup")
    plt.xlabel("Number of CPU cores")
    plt.ylabel("Speedup")
    plt.title("Speedup vs CPU cores")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Chunk sweeping for 12 workers (found to be optimal most of the time on my machine)
    print("\nChunk sweep for 12 workers:")
    n_workers = 12
    chunks_list = []
    times_list = []
    lif_list = []
    for mult in [1, 2, 4, 8, 16, 32]:
        n_chunks = mult * n_workers
        chunk_size = max(1, N // n_chunks)
        chunks, row = [], 0
        while row < N:
            end = min(row + chunk_size, N)
            chunks.append((row, end, N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER))
            row = end
        with Pool(processes=n_workers) as pool:
            pool.map(_worker, tiny)
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                np.vstack(pool.map(_worker, chunks))
                times.append(time.perf_counter() - t0)
        t_par = statistics.median(times)
        lif = n_workers * t_par / t_serial - 1
        chunks_list.append(n_chunks)
        times_list.append(t_par)
        lif_list.append(lif)
        print(
            f"{n_chunks:4d} chunks {t_par:.3f}s {t_serial / t_par:.1f}x LIF={lif:.2f}"
        )

    plt.figure(figsize=(8, 6))
    plt.plot(chunks_list, times_list, marker="o")
    plt.xlabel("Number of chunks")
    plt.ylabel("Time (s)")
    plt.title("Chunk sweep")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(chunks_list, lif_list, marker="o")
    plt.xlabel("Number of chunks")
    plt.ylabel("LIF")
    plt.title("Load Imbalance Factor vs chunks")
    plt.grid(True)
    plt.show()

    # dask cluster test
    print("\nDask chunk sweep:")
    chunks_list_dask = []
    times_list_dask = []
    lif_list_dask = []

    cluster = LocalCluster(n_workers=12, threads_per_worker=1)
    client = Client(cluster)
    # client=Client("tcp://10.92.1.203:8786")
    client.run(
        lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10)
    )  # warm-up
    for mult in [1, 2, 4, 8, 16, 32]:
        n_chunks = mult * n_workers
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            mandelbrot_dask(X_MIN, X_MAX, Y_MIN, Y_MAX, N, MAX_ITER, n_chunks)
            times.append(time.perf_counter() - t0)
        t_par = statistics.median(times)
        lif = n_workers * t_par / t_serial - 1
        chunks_list_dask.append(n_chunks)
        times_list_dask.append(t_par)
        lif_list_dask.append(lif)
        print(
            f"{n_chunks:4d} chunks {t_par:.3f}s {t_serial / t_par:.1f}x LIF={lif:.2f}"
        )

    plt.figure(figsize=(8, 6))
    plt.plot(chunks_list_dask, times_list_dask, marker="o")
    plt.xlabel("Number of chunks")
    plt.ylabel("Time (s)")
    plt.title("Chunk sweep (dask)")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(chunks_list_dask, lif_list_dask, marker="o")
    plt.xlabel("Number of chunks")
    plt.ylabel("LIF")
    plt.title("Load Imbalance Factor vs chunks (dask)")
    plt.grid(True)
    plt.show()
    client.close()
    cluster.close()
