# Interface
## Overview
This is the main interface for our package. This class supports differentiating a function with respect to a point or a list of points. In addition, this class also can differentiate a multivariable function and return a jacobian matrix.


## Usage
After instantiating the interface by passing in a function, the user can pass in a scalar or a list of numbers into the get_der method to evaluate the derivative(s) of the function with respect to the point(s). In the multivariable case, this function will return the jacobian matrix.

Scalar function case:

```python
>>> import AutoDiff
>>> def square_fn(x):
...	return x ** 2
>>> ad_square = AutoDiff(square_fn)
>>> ad_square.get_der(3)
6
```

Vector function case:
```python
>>> import AutoDiff
>>> def square_fn(x):
...	return x ** 2
>>> ad_square = AutoDiff(square_fn)
>>> ad_square.get_der([1,2])
np.array([2,4])
```

Multivariable case:
