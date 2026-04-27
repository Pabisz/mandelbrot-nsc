"""
GPU accelerated Mandelbrot Set Generator

Author: [Sebastian Pabisz Frolund]
Course: Numerical Scientific Computing 2026
"""

import pyopencl as cl
import numpy as np
import time, matplotlib.pyplot as plt
import statistics

KERNEL_F32 = """
__kernel void mandelbrot_f32(
    __global int *result,
    const float x_min, const float x_max,
    const float y_min, const float y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= N || row >= N) return;

    float c_real = x_min + col * (x_max - x_min) / (float)N;
    float c_imag = y_min + row * (y_max - y_min) / (float)N;

    float z_real = 0.0f, z_imag = 0.0f;
    int count = 0;
    while (count < max_iter && z_real*z_real + z_imag*z_imag <= 4.0f) {
        float tmp = z_real*z_real - z_imag*z_imag + c_real;
        z_imag = 2.0f * z_real * z_imag + c_imag;
        z_real = tmp;
        count++;
    }
    result[row * N + col] = count;
}

"""

KERNEL_F64 = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void mandelbrot_f64(
    __global int *result,
    const double x_min, const double x_max,
    const double y_min, const double y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= N || row >= N) return;

    double c_real = x_min + col * (x_max - x_min) / (double)N;
    double c_imag = y_min + row * (y_max - y_min) / (double)N;

    double z_real = 0.0, z_imag = 0.0;
    int count = 0;
    while (count < max_iter && z_real*z_real + z_imag*z_imag <= 4.0) {
        double tmp = z_real*z_real - z_imag*z_imag + c_real;
        z_imag = 2.0 * z_real * z_imag + c_imag;
        z_real = tmp;
        count++;
    }
    result[row * N + col] = count;
}
"""

def mandelbrot_gpu_f32(ctx: cl.Context, queue: cl.CommandQueue, N: int, x_min: float, x_max: float, y_min: float, y_max: float, max_iter: int, timing: bool=False)-> np.ndarray:
    """
    Compute the Mandelbrot set on the GPU using a float32 OpenCL kernel,
    or benchmark kernel execution time.

    Parameters
    ----------
    ctx : cl.Context
        OpenCL context containing the target device.
    queue : cl.CommandQueue
        Command queue used to execute kernels.
    N : int
        Grid resolution (produces an N×N output image).
    x_min : float
        Minimum real-axis value.
    x_max : float
        Maximum real-axis value.
    y_min : float
        Minimum imaginary-axis value.
    y_max : float
        Maximum imaginary-axis value.
    max_iter : int
        Maximum number of iterations for escape-time computation.
    timing : bool, optional
        If False (default), returns the computed Mandelbrot iteration counts.
        If True, returns the median execution time over 3 runs.

    Returns
    -------
    np.ndarray or float
        If timing=False:
            2D array of shape (N, N) containing escape iteration counts.
        If timing=True:
            Median execution time (in seconds) over 3 runs.
    """
    prog = cl.Program(ctx, KERNEL_F32).build()
    result_host = np.zeros((N, N), dtype=np.int32)
    result_dev  = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, result_host.nbytes)
    
    if timing==False:
        prog.mandelbrot_f32(
            queue, (N, N), None, result_dev,
            np.float32(x_min), np.float32(x_max),
            np.float32(y_min), np.float32(y_max),
            np.int32(N), np.int32(max_iter))
        cl.enqueue_copy(queue, result_host, result_dev)
        queue.finish()
        return result_host
    
    else:
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            prog.mandelbrot_f32(
                queue, (N, N), None, result_dev,
                np.float32(x_min), np.float32(x_max),
                np.float32(y_min), np.float32(y_max),
                np.int32(N), np.int32(max_iter))
            cl.enqueue_copy(queue, result_host, result_dev)
            queue.finish()
            times.append(time.perf_counter() - t0)
        return statistics.median(times)


def mandelbrot_gpu_f64(ctx: cl.Context, queue: cl.CommandQueue, N: int, x_min: float, x_max: float, y_min: float, y_max: float, max_iter: int, timing: bool=False)-> np.ndarray:
    """
    Compute the Mandelbrot set on the GPU using a float64 OpenCL kernel,
    or benchmark kernel execution time.

    Parameters
    ----------
    ctx : cl.Context
        OpenCL context containing the target device.
    queue : cl.CommandQueue
        Command queue used to execute kernels.
    N : int
        Grid resolution (produces an N×N output image).
    x_min : float
        Minimum real-axis value.
    x_max : float
        Maximum real-axis value.
    y_min : float
        Minimum imaginary-axis value.
    y_max : float
        Maximum imaginary-axis value.
    max_iter : int
        Maximum number of iterations for escape-time computation.
    timing : bool, optional
        If False (default), returns the computed Mandelbrot iteration counts.
        If True, returns the median execution time over 3 runs.

    Returns
    -------
    np.ndarray or float or None
        If timing=False:
            2D array of shape (N, N) with escape iteration counts.
        If timing=True:
            Median execution time (seconds) over 3 runs.
        If the device does not support double precision:
            None
    """
    dev = ctx.devices[0]
    if 'cl_khr_fp64' not in dev.extensions:
        print(f"  {dev.name} does not support float64 (cl_khr_fp64 missing).")
        return None

    prog = cl.Program(ctx, KERNEL_F64).build()
    result_host = np.zeros((N, N), dtype=np.int32)
    result_dev  = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, result_host.nbytes)
    if timing==False:
        prog.mandelbrot_f64(
            queue, (N, N), None, result_dev,
            np.float64(x_min), np.float64(x_max),
            np.float64(y_min), np.float64(y_max),
            np.int32(N), np.int32(max_iter))
        cl.enqueue_copy(queue, result_host, result_dev)
        queue.finish()
        return result_host
    else:
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            prog.mandelbrot_f64(
                queue, (N, N), None, result_dev,
                np.float64(x_min), np.float64(x_max),
                np.float64(y_min), np.float64(y_max),
                np.int32(N), np.int32(max_iter))
            cl.enqueue_copy(queue, result_host, result_dev)
            queue.finish()
            times.append(time.perf_counter() - t0)
        return statistics.median(times)


if __name__ == "__main__":
    N,MAX_ITER,X_MAX,X_MIN,Y_MAX,Y_MIN = 4096, 100, 1.0, -2.0, 1.5, -1.5
    RUNS = 3
    
    ctx   = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    dev   = ctx.devices[0]
    print(f"Device: {dev.name}\n")
    
    # Warm-up pass (triggers kernel compilation)
    _ = mandelbrot_gpu_f32(ctx, queue, N=64, x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX, max_iter=MAX_ITER)

    print(f"Benchmarking N={N}, max_iter={MAX_ITER}, {RUNS} runs each:\n")

    # --- float32 ---
    t32 = mandelbrot_gpu_f32(ctx, queue, N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER, timing=True)
    img_f32 = mandelbrot_gpu_f32(ctx, queue, N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER)
    print(f"  float32: {t32*1e3:.1f} ms")

    # --- float64 ---
    t64 = None
    img_f64 = mandelbrot_gpu_f64(ctx, queue, N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER)
    if img_f64 is not None:
        t64 = mandelbrot_gpu_f64(ctx, queue, N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER, timing=True)
        print(f"  float64: {t64*1e3:.1f} ms")
        print(f"  Ratio float64/float32: {t64/t32:.2f}x")
        diff = np.abs(img_f32.astype(int) - img_f64.astype(int))
        print(f"  Max pixel difference (f32 vs f64): {diff.max()}")

    # --- Visualise ---
    fig, axes = plt.subplots(1, 2 if img_f64 is not None else 1,
                             figsize=(12 if img_f64 is not None else 6, 5))
    if img_f64 is None:
        axes = [axes]

    axes[0].imshow(img_f32, cmap='hot', origin='lower')
    axes[0].set_title(f"float32  ({t32*1e3:.1f} ms)")
    axes[0].axis('off')

    if img_f64 is not None:
        axes[1].imshow(img_f64, cmap='hot', origin='lower')
        axes[1].set_title(f"float64  ({t64*1e3:.1f} ms)")
        axes[1].axis('off')

    plt.suptitle(f"GPU Mandelbrot  N={N}  device: {dev.name}", fontsize=10)
    plt.tight_layout()
    plt.show()