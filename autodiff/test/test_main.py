# from AutoDiff import AutoDiff
from autodiff.interface.interface import AutoDiff
from autodiff.dual.dual import Dual
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

# test for cosine
def cos_fn(x): # trial cosine function to feed into AutoDiff
    return admath.cos(-3*x) + 3*x

ad_cos = AutoDiff(cos_fn)
print(ad_cos.get_der([2,3]))

#
##print(math.sin([4,5]))
## Evaluates derivative of sin(5x)+2x^2 when x = 2, 5
#
#a = np.array([3,4,5])
#a = a*2
#
#
#print(sin_fn(4))
