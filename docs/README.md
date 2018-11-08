
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
>>> import AutoDiff
>>> def square_fn(x):
...    return x ** 2
>>> ad_square = AutoDiff(square_fn)
>>> ad_square.get_der(3)
6
```

Vector function case:
```python
>>> import AutoDiff
>>> def square_fn(x):
...    return x ** 2
>>> ad_square = AutoDiff(square_fn)
>>> ad_square.get_der([1,2])
np.array([2,4])
```

In cases where the user wants to use operations such as sin/cos, they should call those functions from the AutoDiff library so that the derivative can be automatically computed.

SINE, COSINE, EXPONENTIAL function case:
```python
>>> import AutoDiff
>>> def sin_fn(x):
...    return Autodiff.sin(x)
>>> ad_sin = AutoDiff(sin_fn)
>>> ad_sin.get_der(0)
1
```

# Background
Automatic differentiation breaks down any function into its elementary functions using a graph structure and calculates the derivative while retaining the function by using dual numbers. This is accomplished by substituting (x + ɛ x-prime) for x in f(x).

As the steps of the graph structure become successively more complex, the derivatives of the preceding steps are used to compute the derivatives.

The chain rule is important for increasing the robustness of the automatic differentiation class, especially because it allows for the class to calculate the derivative of compositions (which are an important part of approximating non-linear functions).

# Software organization
- High-level overview of how the software is organized.
  * Directory structure
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
