"""
Mandelbrot Set Generator

Author: [Sebastian Pabisz Frolund]
Course: Numerical Scientific Computing 2026
"""
import cProfile , pstats
from mandelbrot import mandelbrot_set_naive , mandelbrot_set_vectorized

cProfile.run ("mandelbrot_set_naive ( -2 , 1, -1.5 , 1.5 , 512 , 512, 100)", "naive_profile.prof")
cProfile.run ("mandelbrot_set_vectorized ( -2 , 1, -1.5 , 1.5 , 512 , 512, 100)", 'vectorized_profile.prof')

for name in ('naive_profile.prof', 'vectorized_profile.prof'):
    stats = pstats.Stats(name)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
    