"""
Parallelized Mandelbrot Set Generator

Author: [Sebastian Pabisz Frolund]
Course: Numerical Scientific Computing 2026
"""
import numpy as np
from numba import njit
from multiprocessing import Pool
import time, psutil, os, statistics, matplotlib.pyplot as plt
from pathlib import Path

@njit(cache=True)
def mandelbrot_pixel(c_real, c_imag, max_iter =100) :
    z_real = z_imag = 0.0
    for n in range ( max_iter ) :
        zr2 = z_real * z_real
        zi2 = z_imag * z_imag
        if zr2 + zi2 > 4.0: return n
        z_imag, z_real = 2.0 * z_real * z_imag + c_imag, zr2 - zi2 + c_real
    return max_iter

@njit(cache=True)
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

def _worker(args):
    return mandelbrot_chunk(*args)

def mandelbrot_parallel(x_min, x_max, y_min, y_max, N, max_iter, n_workers=4, n_chunks=None, pool=None):
    if n_chunks is None:
        n_chunks = n_workers
    
    chunk_size = max(1, N // n_chunks)
    chunks,row = [], 0
    while row < N:
        end = min(row + chunk_size, N)
        chunks.append((row, end, N, x_min, x_max, y_min, y_max, max_iter))
        row = end
    if pool is not None:
        return np.vstack(pool.map(_worker, chunks))
    tiny = [(0, 8, 8, x_min, x_max, y_min, y_max, max_iter)]
    with Pool(processes=n_workers) as pool:
        pool.map(_worker, tiny) # warm-up: Numba JIT in all workers
        return np.vstack(pool.map(_worker, chunks))
    
if __name__ == "__main__":

    N, MAX_ITER = 1024, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2, 1, -1.5, 1.5

    result = mandelbrot_parallel(X_MIN, X_MAX,Y_MIN, Y_MAX,N, MAX_ITER)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(result, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], cmap="viridis",origin="lower",aspect="equal")

    ax.set_xlabel("Re(c)")
    ax.set_ylabel("Im(c)")
    ax.set_title("Parallel Mandelbrot")

    plt.show()

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_serial(X_MIN, X_MAX, Y_MIN, Y_MAX,N, MAX_ITER)
        times.append(time.perf_counter() - t0)
        t_serial = statistics.median(times)
    
    workers_list = []
    efficiency_list = []
    speedup_list = []
    tiny = [(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER)]
    
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
        effeciency = speedup / n_workers * 100
        workers_list.append(n_workers)
        speedup_list.append(speedup)
        efficiency_list.append(effeciency)
        print(f"{n_workers:2d} workers: {t_par:.3f}s, speedup={speedup:.2f}x, eff={effeciency:.0f}%")
        
    plt.figure(figsize=(8,6))
    plt.plot(workers_list, speedup_list, marker="o", label="Measured speedup")
    plt.plot(workers_list, workers_list, "--", label="Ideal speedup")
    plt.xlabel("Number of CPU cores")
    plt.ylabel("Speedup")
    plt.title("Speedup vs CPU cores")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Chunk sweeping for 13 workers (found to be optimal most of the time on my machine)
    n_workers = 13
    chunks_list = []
    times_list = []
    lif_list = []
    for mult in [1,2,4,8,16,32]:
        n_chunks = mult*n_workers
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
        print(f"{n_chunks:4d} chunks {t_par:.3f}s {t_serial/t_par:.1f}x LIF={lif:.2f}")
        
    plt.figure(figsize=(8,6))
    plt.plot(chunks_list, times_list, marker="o")
    plt.xlabel("Number of chunks")
    plt.ylabel("Time (s)")
    plt.title("Chunk sweep")
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(8,6))
    plt.plot(chunks_list, lif_list, marker="o")
    plt.xlabel("Number of chunks")
    plt.ylabel("LIF")
    plt.title("Load Imbalance Factor vs chunks")
    plt.grid(True)
    plt.show()