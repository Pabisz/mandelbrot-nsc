"""
Parallelized Mandelbrot Test Suite

Author: [Sebastian Pabisz Frolund]
Course: Numerical Scientific Computing 2026
"""
import numpy as np
import pytest

from mandelbrot_parallel import (
    mandelbrot_pixel,
    mandelbrot_serial,
    mandelbrot_parallel,
)

def test_origin():
    assert mandelbrot_pixel(0.0, 0.0, 100) == 100

def test_far_outside():
    assert mandelbrot_pixel(5.0, 0.0, 100) == 1

def test_boundary_like_point():
    # kendt tricky punkt
    result = mandelbrot_pixel(-2.0, 0.0, 100)
    assert result > 0

CASES = [
    (0.0, 0.0, 100, 100),
    (5.0, 0.0, 100, 1),
    (-2.5, 0.0, 100, 1),
]

@pytest.mark.parametrize("cr, ci, max_iter, expected", CASES)
def test_pixel_param(cr, ci, max_iter, expected):
    assert mandelbrot_pixel(cr, ci, max_iter) == expected

def test_parallel_equals_serial():
    N = 32
    max_iter = 50
    
    serial = mandelbrot_serial(-2, 1, -1.5, 1.5, N, max_iter)
    parallel = mandelbrot_parallel(-2, 1, -1.5, 1.5, N, max_iter, n_workers=2)

    np.testing.assert_array_equal(serial, parallel)