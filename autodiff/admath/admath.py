from autodiff.dual.dual import Dual as Dual
import math

def sin(x):
    if (isinstance(x,Dual)):
        x.der = math.cos(x.val)*x.der
        x.val = math.sin(x.val)
        return x
    else:
        return math.sin(x)
    
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

def loge(x):
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

def log10(x):
    """Calculate the log10 of the input

        Keyword arguments:
        x -- a real number or a dual number

        Return:
        the log10 value
    """
    if (isinstance(x,Dual)):
        x.der = (1/x.val)*x.der
        x.val = np.log10(x.val)
        return x
    else:
        return np.log10(x)


def log2(x):
    """Calculate the log2 of the input

        Keyword arguments:
        x -- a real number or a dual number

        Return:
        the log2 value
    """
    if (isinstance(x,Dual)):
        x.der = (1/x.val)*x.der
        x.val = np.log2(x.val)
        return x
    else:
        return np.log2(x)


def exp(x):
    """Calculate the exponential of the input

        Keyword arguments:
        x -- a real number or a dual number

        Return:
        the exponential value
        """
    if (isinstance(x,Dual)):
        x.der = x.val * np.log(x.val) *x.der
        x.val = np.exp(x.val)
        return x
    else:
        return np.exp(x)


def sqrt(x):
    """Calculate the square root of the input

        Keyword arguments:
        x -- a real number or a dual number

        Return:
        the exponential value
        """
    if (isinstance(x,Dual)):
        x.der = 0.5/np.sqrt(x.val) * x.der
        x.val = np.sqrt(x.val)
        return x
    else:
        return np.sqrt(x)
