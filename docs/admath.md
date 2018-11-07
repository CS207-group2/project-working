# admath

## Overview
This module performs both value and derivative calculations of elemental functions, such as trig and exponential, for the dual class. We use the same function names as numpy to allow for easier usability. The functions in this module also work with scalars.

## Usage
This library is used when the function that we want to evaluate a derivate of has functions such as sin/cos/log. Inside the function that will be passed into the interface, all the elemental functions should be called using this library as shown in the example below. This is so that the dual class can keep track of all the derivatives.


```python
>>> import autodiff
>>> def sin_fn(x):
...	return autodiff.sin(x)
>>> ad_sin = autodiff(sin_fn)
>>> ad_sin.get_der(0)
1
```

## List of Functions

sin </br>
cos </br>
etc.....
