from Dual.dual import Dual
import math

def sin(x):
    if (isinstance(x,Dual)):
        x.der = math.cos(x.val)*x.der        
        x.val = math.sin(x.val)
        return x
    else:
        return math.sin(x)
