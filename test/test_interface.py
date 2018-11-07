import pytest
from autodiff.interface.interface import AutoDiff as AD
import autodiff.admath.admath as math

def square_fn(x):
        return x ** 2
    
def test_square_fn():
    ad_square = AD(square_fn)
    der = ad_square.get_der(3)
    assert der == 6