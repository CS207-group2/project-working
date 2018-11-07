import pytest
from autodiff.dual import dual
import autodiff.admath.admath as math
import numpy as np

def sin_fn(x):
    return math.sin(x)

def test__sin_fn__():
    x = dual.Dual(0)
    f = sin_fn(x)
    assert f.val == 0
    assert f.der == 1
    
    
