# AutoDiff.Dual

## Implementation of unary operations
We implemented the following unary operations:
``__neg__`` and ``__pos__``.

## Implementation of binary operations

We provide full implementations off all binary differentiable operations with dual and non-dual numbers. When we want to operate a with $u$, $ a \circ u$, we have 4 possibities: 

1. $a$ and $u$ are both scalars
2. $a$ is a dual number, $b$ is a scalar
3. $a$ is a scalar, $b$ is a dual number
4. $a$ and $u$ are both dula numbers

Operations 1 is already implemented by Python, hence we only need to implement operations 2 through 4.

In the class AutoDiff.Dual we will implement the following binary operations:
* Addition
* Substration
* Multiplication
* Division
* Power

Below are the derivative formula for each operation. Dual numbers are marked as a function with a variable, e.g. $u(x)$ or $a(x)$ with ther derivative, $a^{\prime}$ and $u^{\prime}$. Scalars are simply labeled without variable: $a$ and $b$. 

### Addition operation
Overload `__add__` to account for:
$$ \dfrac{d}{dx}\Big[a(x) + u\Big]= a^\prime $$
$$ \dfrac{d}{dx}\Big[a(x) + u(x)\Big]= a^\prime + u^\prime$$
Overload `__radd__` to account for:
$$ \dfrac{d}{dx}\Big[a + u(x)\Big]= u^\prime $$

### Substration operation
Overload `__sub__` to account for:
$$ \dfrac{d}{dx}\Big[a(x) - u\Big]= a^\prime $$
$$ \dfrac{d}{dx}\Big[a(x) - u(x)\Big]= a^\prime - u^\prime$$
Overload `__rsub__` to account for:
$$ \dfrac{d}{dx}\Big[a - u(x)\Big]= - u^\prime $$


### Multiplication operation
Overload `__mul__` to account for:
$$ \dfrac{d}{dx}\Big[a(x)u\Big]= u a^\prime $$
$$ \dfrac{d}{dx}\Big[a(x)u(x)\Big]= a^\prime u + a u^\prime$$
Overload `__rmul__` to account for:
$$ \dfrac{d}{dx}\Big[au(x)\Big]= a u^\prime $$


### Division operation
Overload `__truediv__` to account for:
$$ \dfrac{d}{dx}\Big[\dfrac{a(x)}{u}\Big]= \dfrac{a^\prime}{u} $$
$$ \dfrac{d}{dx}\Big[\dfrac{a(x)}{u(x)}\Big]= -\dfrac{a^\prime u - a u^\prime}{{u(x)}^2}$$
Overload `__rtruediv__` to account for:
$$ \dfrac{d}{dx}\Big[\dfrac{a}{u(x)}\Big]= -\dfrac{a}{{u(x)}^2} u^\prime $$


### Power operation
Overload `__pow__` to account for:
$$ \dfrac{d}{dx} a(x)^n = n a^{n-1} a^{\prime}$$
$$ \dfrac{d}{dx} a(x)^{u(x)} = a^{u-1} \Big[u a^\prime + a \log(a) u^\prime \Big]$$
Overload `__rpow__` to account for:
$$ \dfrac{d}{dx} a^{u(x)} = log(a) a^{u(x)} u^\prime $$
