import pytest
from autodiff.dual import dual
import autodiff.admath.admath as math
import numpy as np

def test__sin_fn__():
    x = dual.Dual(0)
    f = math.sin(x)
    assert f.val == 0
    assert f.der == 1
    
    
