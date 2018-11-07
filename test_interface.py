import pytest
from AutoDiff import *

def square_fn(x):
        return x ** 2
    
def test_square_fn():
    ad_square = AD(square_fn)
    der = ad_square.get_der(3)
    assert der == 6

def test_get_der_types():
    with pytest.raises(TypeError):
        AutoDiff.get_der('string')
    with pytest.raises(TypeError):
        AutoDiff.get_der(dict[1:'a', 2:'b'])

def test_get_der_lenlist():
    a = AutoDiff(lambda x,y: 3*x**2 + 2*y**3)
    with pytest.raises(Exception):
        a.get_der(1, 2, 3)
    with pytest.raises(Exception):
        a.get_der([1, 2, 3], [3, 4, 5], [1, 3, 4])
