from autodiff.dual.dual import Dual
import inspect

class AutoDiff:
    def __init__(self, fn):
        self.fn = fn

    def get_der(self, val):
        """ Returns derivatives of the function evaluated at values given.
        
        inputs
        ------
        val : single number, a list of numbers, or a list of lists

        returns
        ------
        derivates in the same shape given
        """
        ders = []
        sig = inspect.signature(self.fn)
        l = len(list(sig.parameters))
        if l >= 2:
            #for list of lists, each list evaluated at different variables
            if any(isinstance(el, list) for el in val) is True:  
                list_der = []
                for p in val:
                    list_der.append(self.get_der(p))
                return list_der
            elif l != len(val):
                raise Exception('Function requires {} values that correspond to the multiple variables'.format(l))
            else:
                #for a list of numbers, evaluated at different variables.
                for i in range(l): 
                    new_val = val.copy()
                    new_val[i] = Dual(new_val[i])
                    ders.append(self.fn(*new_val).der)
                return ders
        #for a list of numbers, evaluated at a single variable.
        if (isinstance(val,list)): 
            for v in val:
                a = Dual(v)
                ders.append(self.fn(a).der)
            return ders
        else:
            a = Dual(val)
            return self.fn(a).der


       
