import numpy as np

class Dual:
    
    def __init__(self, x, der=1):
        self.val = x
        self.der = der
        
    ## UNARY OPERATIONS
        
        
    def __neg__(self):
        return Dual(-self.val, -self.der)
    
    def __pos__(self):
        return Dual(+self.val, +self.der)
        
        
    ## PLUS OPERATIONS
    
    def __add__(self, other):
        try:
            return Dual(self.val + other.val, self.der + other.der)
        except AttributeError:
            return Dual(self.val + other, self.der)
    
    def __radd__(self, other):
        return Dual(other + self.val, self.der)
    
    
    ## MINUS OPERATIONS
    
    def __sub__(self, other):
        try:
            return Dual(self.val - other.val, self.der - other.der)
        except AttributeError:
            return Dual(self.val - other, self.der)
        
    def __rsub__(self, other):
        return Dual(other - self.val, -self.der)
        
    
    ## MULTIPLICATION OPERATIONS
    
    def __mul__(self, other):
        try:
            # multiplication rule
            temp = self.val * other.der + self.der * other.val
            return Dual(self.val * other.val, temp)
        except AttributeError:
            return Dual(self.val * other, self.der * other)

    def __rmul__(self, other):
        return Dual(self.val * other, self.der * other)
    
    
    ## DIVISION OPERATIONS
        
    def __truediv__(self, other):
        try:
            # quotient rule
            temp = (self.der * other.val - self.val * other.der)
            print(self.der)
            print(other.der)
            return Dual(self.val/other.val, temp/other.val ** 2)
        except AttributeError:
            # divide by a constant
            return Dual(self.val/other, self.der/other)
        
    def __rtruediv__(self, other):
            return Dual(other/self.val, -other/self.val**2*self.der)   
    
    
    ## POWER OPERATIONS
    
    def __pow__(self, other):
        try:
            # da^u/dx = ln(a) a^u du/dx
            factor = self.val ** (other.val -1)
            sum_1 = other.val * self.der
            sum_2 = self.val * np.log(self.val) * other.der
            temp = factor * (sum_1 + sum_2)
            return Dual(self.val ** other.val, temp)
        except AttributeError:
            # du^n/dx = n * u^(n-1) * du/dx
            temp = other * self.val ** (other-1) * self.der
            return Dual(self.val ** other, temp)
        
    def __rpow__(self, other):
            print("__rpow__")
            # da^u/dx = ln(a) a^u du/dx
            temp = np.log(other) * other ** self.val * self.der
            return Dual(other ** self.val, temp)
        