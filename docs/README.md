
# Introduction
Automatic differentiation is a set of techniques to numerically evaluate the derivative of a function specified by a computer program. Automatic differentiation breaks down a function by looking at the sequence of elementary arithmetic operations (addition, subtraction, multiplication and division) and elementary functions (exponential, log10, log2, loge, sin, cos, etc). By applying the chain rule repeatedly to these operations, derivatives of arbitrary order can be computed automatically, accurately to machine accuracy. The major application of automatic differentiation is gradient-based optimization, which is commonly used as the foundation of neural nets.

This package, `autodiff`, is a package of automatic differentiation, which means it can automatically differentiate a function input into the program.

The package currently supports forward-mode differentiation, which means the chain rule is traversed from inside to outside.

# How to use your package

<!-- - How to install?  Even (especially) if the package isn't on `PyPI`, you should walk them through
the creation of a virtual environment or some other kind of manual installation.
- Include a basic demo for the user.  Come up with a simple function to differentiate and walk the
user through the steps needed to accomplish that task. -->

## How to install?
Currently, a user can install the package by following the steps below:
1. Go to your project directory and create a virtual environment `python3 -m virtualenv env`
* you might need to download `virtualenv` if you do not have it.
2. Download the package from our GitHub [repository](https://github.com/CS207-group2/cs207-FinalProject/) to your virtual environment
<!-- 3. Move the package to the virtual environment -->
3. run `python setup.py install`

## How to use *Autodiff*?
The user can use AutoDiff by passing a function to the AutoDiff constructor to create an AutoDiff object. Then, the user can evaluate the derivative of that function at a certain value by passing in that value to the object. This object can then be called to return the derivative of the function evaluated at that point.

Scalar function case:
```python
>>> from autodiff.interface.interface import AutoDiff as AD
>>> def square_fn(x):
...    return x ** 2
>>> ad_square = AD(square_fn)
>>> ad_square.get_der(3)
6
```

Vector function case:
```python
>>> from autodiff.interface.interface import AutoDiff as AD
>>> def square_fn(x):
...    return x ** 2
>>> ad_square = AD(square_fn)
>>> ad_square.get_der([1,2])
np.array([2,4])
```

In cases where the user wants to use operations such as sin/cos, they should call those functions from the AutoDiff library so that the derivative can be automatically computed.

SINE, COSINE, EXPONENTIAL function case:
```python
>>> from autodiff.interface.interface import AutoDiff as AD
>>> import autodiff.admath.admath as admath

>>> def sin_fn(x):
...    return AD.sin(x)
>>> ad_sin = AutoDiff(sin_fn)
>>> ad_sin.get_der(0)
1
```

# Background
Automatic differentiation breaks down any function into its elementary functions using a graph structure, where every node is an operation, and calculates the derivative on top of the numerical value. The simultaneous value and derivative calculation is accomplished by using dual numbers, which are numbers have an additional component ɛ on top of its real component (called dual component).

Dual numbers can simply be used by substituting (x + ɛ x') for x in f(x) where f can be any one operation. The important idea is that after every mathematical operation, the real part will represent the numerical value of the expression and the dual part will reflect the derivative. This property makes it very convenient for derivative calculations of heavily nested functions because of the chain rule in derivative calculation which states that the derivative of f(g(x)) is f'(g(x)) * g'(x). Since the derivative of a nested function relies on both the value and the derivative of the inner function, we can see that the automatic storage of both the value and derivative after every operation is very convenient for this task.



# Software organization
- High-level overview of how the software is organized.
  * Directory structure
  ```
   FinalProject\
         autodiff\
               __init__.py
               admath/
                    __init__.py
                    admath.py
               dual/
                    __init__.py
                    dual.py
               interface/
                    __init__.py
                    interface.py
         test\
              __init__.py
              .coverage
              test_admath.py
              test_dual.py
              test_interface.py
         docs\
              admath.md
              dual.md
              interface.md
              README.md
         README.md
         setup.py
         LICENSE.txt
         .gitignore
         .travis.yml
         setup.cfg
  ```
  * Basic modules and what they do
    * `admath`
      * this is the module for math computation. It leverages numpy library and provides functions including elementary functions (exponential, log10, log2, loge, sin, cos)
    * `interface`
    * `dual`
      * this is the module for the dual number class.
  * Where do tests live?  How are they run?  How are they integrated?
    * Tests of this package are in the `test` folder.
    * They are run by `TravisCI` and the coverage is examined by `Coveralls`
    * We have embedded the badges in the README of the package
  * How can someone install your package?  
    * Currently, the package is available to download from the GitHub repo and the installation details is included in the `How to install?` section
    * We plan to distribute the package through `PyPI` in the near future

# Implementation details
Currently, the autodiff package contains 2 classes and 1 module.

### interface Class
#### Usage
Interface is our main class where an instance of our class can be instantiated by passing in a function. Next, the user can pass in a scalar or a list of numbers into the get_der method to evaluate the derivative(s) of the function with respect to the point(s). Furthermore, our class supports multivariable differentiation, where the user can write a multivariable function, pass in a 2d list where each list represents the derivative calculation at each value, and get back the Jacobian matrix.

#### Implementation
Interface


### dual Class
#### Usage
The dual class represents a dual number. The user does not have to explicitly instantiate it to use our package as it will be instantiated by the Interface class automatically.

#### Implementation
The dual object contains val and der attributes, representing the numerical value and derivative respectively. It contains dunder methods to handle all basic math operations such as add, multiply, power, in cases where both numbers being added are dual numbers (eg a+b where both a and b are dual) as well as in cases where the left or the right side of the expression is a scalar (eg a+b where a is scalar and b is dual).

### admath Module
#### Usage
This admath module performs both value and derivative calculations of elemental functions, such as trig and exponential, for the dual class. We use the same function names as numpy to allow for easier usability for people already used to numpy. The functions in this module also work with scalars.

#### Implementation
We implemented the following functions:
- sin(x)
- cos(x)
- log(x)
- log10(x)
- log2(x)
- exp(x)
- sqrt(x)

In cases where x is a scalar, we simply return the numpy equivalent (eg np.sin(x)). When x is dual, we manually set val and der of the dual object. We set the der by figuring out symbolically what the derivative should be (sin(x) should be cos(x)) and applying the chain rule (multiplying x.der to cos(x)). This way, our program can automatically apply the chain rule to our inputs and handle nested functions with ease. Again, like the scalar case, we use numpy to do the actual elemental calculations.





- Description of current implementation.  This section goes deeper than the high level software
organization section.
* Try to think about the following:
- Core data structures
- Core classes
- Important attributes
- External dependencies
- Elementary functions
This is similar to what you did for milestone 1, but now you've actually implemented it.
- What aspects have you not implemented yet?  What else do you plan on implementing?
