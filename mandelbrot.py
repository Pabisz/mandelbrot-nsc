"""
Mandelbrot Set Generator

Author: [Sebastian Pabisz Frolund]
Course: Numerical Scientific Computing 2026
"""
import numpy as np
import matplotlib.pyplot as plt
import time, statistics
from numba import njit

def mandelbrot_point(c, max_iter):
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z * z + c
    return max_iter

def mandelbrot_set_naive(xmin, xmax, ymin, ymax, width, height, max_iter):
    result = np.zeros((height, width), dtype=int)

    x_values = np.linspace(xmin, xmax, width)
    y_values = np.linspace(ymin, ymax, height)

    for i in range(height):
        for j in range(width):
            c = complex(x_values[j], y_values[i])
            result[i, j] = mandelbrot_point(c, max_iter)

    return result

def mandelbrot_set_vectorized(xmin, xmax, ymin, ymax, width, height, max_iter):
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
def mandelbrot_point_numba (c , max_iter =100) :
    z = 0 + 0j
    for n in range ( max_iter ) :
        if z.real * z.real + z.imag * z.imag > 4.0:
            return n
        z = z *z + c
    return max_iter

def mandelbrot_hybrid(xmin, xmax, ymin, ymax, width, height, max_iter):
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
def mandelbrot_naive_numba(xmin, xmax, ymin, ymax, width, height, max_iter, dtype = np . float64):
    x = np.linspace ( xmin , xmax , width ).astype(dtype)
    y = np.linspace ( ymin , ymax , height ).astype(dtype)
    result = np . zeros (( height , width ), dtype = np.int32)
    for i in range ( height ):
        for j in range ( width ):
            c = x [j] + 1j * y[ i]
            z = 0 + 0j
            n = 0
            while n < max_iter and \
                z.real * z.real + z.imag * z.imag <= 4.0:
                z = z*z + c ; n += 1
            result [i , j ] = n
    return result
    

def bench (fn , * args , runs =5, **kwargs):
    fn (* args , **kwargs) # warm -up
    times = []
    for _ in range ( runs ) :
        t0 = time . perf_counter ()
        fn (* args , **kwargs)
        times . append ( time . perf_counter () - t0 )
    return statistics . median ( times )
mandelbrot_naive_numba ( -2 , 1 , -1.5 , 1.5 , 1024 , 1024 ,100 , dtype = np.float64 )

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
    print(f"Naive implementation took {t_naive:.4f} seconds")
    print(f"Vectorized implementation took {t_vectorized:.4f} seconds")
    print(f"Hybrid implementation took {t_hybrid:.4f} seconds")
    print(f"Numba implementation took {t_numba:.4f} seconds")

    for dtype in [ np.float32 , np.float64 ]:
        t_numba_dtype = bench(mandelbrot_naive_numba, -2 , 1 , -1.5 , 1.5 , 1024 , 1024, 100, dtype=dtype)
        print(f"Numba implementation with {dtype} took {t_numba_dtype:.4f} seconds")
    
    r32 = mandelbrot_naive_numba ( -2 , 1 , -1.5 , 1.5 , 1024 , 1024 ,100 , dtype = np.float32 )
    r64 = mandelbrot_naive_numba ( -2 , 1 , -1.5 , 1.5 , 1024 , 1024 ,100 , dtype = np.float64 )
    fig , axes = plt . subplots (1 , 2, figsize =(12 , 4) )
    for ax , result , title in zip ( axes , [ r32 , r64 ] , ["float32 ", "float64 ( ref )"]) :
        ax.imshow ( result , cmap ="viridis", extent =( xmin , xmax , ymin , ymax ) )
        ax.set_title(title); ax.axis ("off")
    plt . savefig ("precision_comparison.png", dpi =150)
    print (f" Max diff float32 vs float64 : {np.abs(r32 - r64 ). max ()}")
    
        
    # Plot the Mandelbrot set using vectorized implementation
    #mandelbrot_image = mandelbrot_set_vectorized(*args)
    #plt.imshow(mandelbrot_image, extent=(xmin, xmax, ymin, ymax),cmap='viridis')
    #plt.colorbar()
    #plt.title('Mandelbrot Set')
    #plt.xlabel('Real Axis')
    #plt.ylabel('Imaginary Axis')
    #plt.show()
