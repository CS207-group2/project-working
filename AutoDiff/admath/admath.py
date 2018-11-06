from ..dual.dual import Dual
#import math
import numpy as np

def sin(x):
    if (isinstance(x,Dual)):
        x.der = np.cos(x.val)*x.der
        x.val = np.sin(x.val)
        return x
    else:
        return np.sin(x)

def cos(x):
    """Calculate cosine of the input
        
        Keyword arguments:
        x -- a real number or a dual number
        
        Return:
        the cosine value
    """
    if (isinstance(x,Dual)):
        x.der = -1 * np.sin(x.val)*x.der
        x.val = np.cos(x.val)
        return x
    else:
        return np.cos(x)

def log(x):
    """Calculate the natural log of the input
        
        Keyword arguments:
        x -- a real number or a dual number
        
        Return:
        the natural log value
    """
    if (isinstance(x,Dual)):
        x.der = (1/x.val)*x.der
        x.val = np.log(x.val)
        return x
    else:
        return np.log(x)

def exp(x):
    """Calculate the exponential of the input
        
        Keyword arguments:
        x -- a real number or a dual number
        
        Return:
        the exponential value
        """
    if (isinstance(x,Dual)):
        x.der = *x.der
        x.val = np.cos(x.val)
        return x
    else:
        return np.cos(x)
