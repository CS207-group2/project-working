from AutoDiff import AutoDiff
import admath
#from ..dual.dual import Dual

def square_fn(x):
    return x ** 2


def sin_fn(x):
    return admath.sin(5*x)+2*x**2

ad_square = AutoDiff(square_fn)
print(ad_square.get_der(3))

print(ad_square.get_der([1,2]))

ad_sin = AutoDiff(sin_fn)
print(ad_sin.get_der(2))
# Evaluates derivative of sin(5x)+2x^2 when x = 2
