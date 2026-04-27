"""
Mandelbrot Set Generator

Author: [Sebastian Pabisz Frolund]
Course: Numerical Scientific Computing 2026
"""
import numpy as np
import matplotlib.pyplot as plt
import time, statistics
from numba import njit

def mandelbrot_point(c: complex, max_iter: int)-> int:
    """
    Compute the escape iteration count for a single complex point.

    Parameters
    ----------
    c : complex
        Complex parameter defining the Mandelbrot iteration.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    int
        Number of iterations before |z| > 2, or max_iter if bounded.
    """
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z * z + c
    return max_iter

def mandelbrot_set_naive(xmin: float, xmax: float, ymin: float, ymax: float, width: int, height: int, max_iter: int)-> np.ndarray:
    """
    Compute the Mandelbrot set using a naive approach.

    Parameters
    ----------
    xmin : float
        Minimum real-axis value.
    xmax : float
        Maximum real-axis value.
    ymin : float
        Minimum imaginary-axis value.
    ymax : float
        Maximum imaginary-axis value.
    width : int
        Width of the output image.
    height : int
        Height of the output image.
    max_iter : int
        Maximum number of iterations for escape-time computation.

    Returns
    -------
    np.ndarray
        2D array of shape (height, width) containing escape iteration counts.
    """
    result = np.zeros((height, width), dtype=int)

    x_values = np.linspace(xmin, xmax, width)
    y_values = np.linspace(ymin, ymax, height)

    for i in range(height):
        for j in range(width):
            c = complex(x_values[j], y_values[i])
            result[i, j] = mandelbrot_point(c, max_iter)

    return result

def mandelbrot_set_vectorized(xmin: float, xmax: float, ymin: float, ymax: float, width: int, height: int, max_iter: int)-> np.ndarray:
    """
    Compute the Mandelbrot set using a vectorized approach.

    Parameters
    ----------
    xmin : float
        Minimum real-axis value.
    xmax : float
        Maximum real-axis value.
    ymin : float
        Minimum imaginary-axis value.
    ymax : float
        Maximum imaginary-axis value.
    width : int
        Width of the output image.
    height : int
        Height of the output image.
    max_iter : int
        Maximum number of iterations for escape-time computation.

    Returns
    -------
    np.ndarray
        2D array of shape (height, width) containing escape iteration counts.
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

@njit
def mandelbrot_point_numba (c : complex, max_iter : int) -> int:
    """
    Compute escape iteration count for a single point using Numba JIT.

    Parameters
    ----------
    c : complex
        Complex parameter.
    max_iter : int, optional
        Maximum number of iterations (default is 100).

    Returns
    -------
    int
        Escape iteration count.
    """
    z = 0 + 0j
    for n in range ( max_iter ) :
        if z.real * z.real + z.imag * z.imag > 4.0:
            return n
        z = z *z + c
    return max_iter

def mandelbrot_hybrid(xmin: float, xmax: float, ymin: float, ymax: float, width: int, height: int, max_iter: int)-> np.ndarray:
    """
    Compute the Mandelbrot set using Python loops with a Numba-accelerated point function.

    Parameters
    ----------
    xmin, xmax : float
        Real-axis bounds.
    ymin, ymax : float
        Imaginary-axis bounds.
    width : int
        Number of x-axis samples.
    height : int
        Number of y-axis samples.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    np.ndarray
        2D array of escape iteration counts.
    """
    x = np . linspace ( xmin , xmax , width )
    y = np . linspace ( ymin , ymax , height )
    result = np . zeros (( height , width ) ,
    dtype = np . int32 )
    for i in range ( height ):
        for j in range ( width ):
            c = x [j] + 1j * y[ i]
            result [i ,j] = mandelbrot_point_numba(c, max_iter)
    return result

@njit
def mandelbrot_naive_numba(xmin: float, xmax: float, ymin: float, ymax: float, width: int, height: int, max_iter: int, dtype = np . float64)-> np.ndarray:
    """
    Fully Numba-accelerated Mandelbrot computation using float64 precision.

    Parameters
    ----------
    xmin, xmax : float
        Real-axis bounds.
    ymin, ymax : float
        Imaginary-axis bounds.
    width : int
        Grid width.
    height : int
        Grid height.
    max_iter : int
        Maximum number of iterations.
    dtype : np.dtype, optional
        Floating-point precision (default is float64).

    Returns
    -------
    np.ndarray
        2D array of escape iteration counts (int32).
    """
    x = np.linspace ( xmin , xmax , width ).astype(dtype)
    y = np.linspace ( ymin , ymax , height ).astype(dtype)
    result = np . zeros (( height , width ), dtype = np.int32)
    for i in range ( height ):
        for j in range ( width ):
            c = np.complex128(x[j] + y[i] * 1j)
            z = np.complex128(0.0 + 0.0 * 1j)
            n = 0
            while n < max_iter and \
                z.real * z.real + z.imag * z.imag <= dtype(4.0):
                z = z*z + c ; n += 1
            result [i , j ] = n
    return result

@njit
def mandelbrot_naive_numba32(xmin: float, xmax: float, ymin: float, ymax: float, width: int, height: int, max_iter: int, dtype = np.float32)-> np.ndarray:
    """
    Fully Numba-accelerated Mandelbrot computation using float32 precision.

    Parameters
    ----------
    xmin, xmax : float
        Real-axis bounds.
    ymin, ymax : float
        Imaginary-axis bounds.
    width : int
        Grid width.
    height : int
        Grid height.
    max_iter : int
        Maximum number of iterations.
    dtype : np.dtype, optional
        Floating-point precision (default is float32).

    Returns
    -------
    np.ndarray
        2D array of escape iteration counts (int32).
    """
    x = np.linspace ( xmin , xmax , width ).astype(dtype)
    y = np.linspace ( ymin , ymax , height ).astype(dtype)
    result = np . zeros (( height , width ), dtype = np.int32)
    for i in range ( height ):
        for j in range ( width ):
            c = np.complex64(x[j] + y[i] * 1j)
            z = np.complex64(0.0 + 0.0 * 1j)
            n = 0
            while n < max_iter and \
                z.real * z.real + z.imag * z.imag <= dtype(4.0):
                z = z*z + c ; n += 1
            result [i , j ] = n
    return result

def bench (fn: callable, * args , runs: int =5, **kwargs) -> float:
    """
    Benchmark a function by measuring median execution time.

    Performs one warm-up run, then executes the function multiple times
    and returns the median runtime.

    Parameters
    ----------
    fn : callable
        Function to benchmark.
    *args :
        Positional arguments passed to the function.
    runs : int, optional
        Number of timing runs (default is 5).
    **kwargs :
        Keyword arguments passed to the function.

    Returns
    -------
    float
        Median execution time in seconds.
    """
    fn (* args , **kwargs) # warm -up
    times = []
    for _ in range ( runs ) :
        t0 = time . perf_counter ()
        fn (* args , **kwargs)
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
    t_hybrid = bench(mandelbrot_hybrid, *args)
    t_numba = bench(mandelbrot_naive_numba, *args)
    t_numba32 = bench(mandelbrot_naive_numba32, *args)
    print(f"Naive implementation took {t_naive:.4f} seconds")
    print(f"Vectorized implementation took {t_vectorized:.4f} seconds")
    print(f"Hybrid implementation took {t_hybrid:.4f} seconds")
    print(f"Numba implementation took {t_numba:.4f} seconds")
    print(f"Numba32 implementation took {t_numba32:.4f} seconds")
    
    r32 = mandelbrot_naive_numba32 ( -2 , 1 , -1.5 , 1.5 , 1024 , 1024 , 100)
    r64 = mandelbrot_naive_numba ( -2 , 1 , -1.5 , 1.5 , 1024 , 1024 ,100 )
    fig , axes = plt . subplots (1 , 2, figsize =(12 , 4) )
    for ax , result , title in zip ( axes , [ r32 , r64 ] , ["float32 ", "float64 ( ref )"]) :
        ax.imshow ( result , cmap ="viridis", extent =( xmin , xmax , ymin , ymax ) )
        ax.set_title(title); ax.axis ("off")
    plt . savefig ("precision_comparison.png", dpi =150)
    plt.show()
    print (f" Max diff float32 vs float64 : {np.abs(r32 - r64 ). max ()}")
    print("Median diff:", np.median(np.abs(r32 - r64 )))
    print("Number of differing pixels:", np.count_nonzero(np.abs(r32 - r64 )))
    
        
    # Plot the Mandelbrot set using vectorized implementation
    #mandelbrot_image = mandelbrot_set_vectorized(*args)
    #plt.imshow(mandelbrot_image, extent=(xmin, xmax, ymin, ymax),cmap='viridis')
    #plt.colorbar()
    #plt.title('Mandelbrot Set')
    #plt.xlabel('Real Axis')
    #plt.ylabel('Imaginary Axis')
    #plt.show()
