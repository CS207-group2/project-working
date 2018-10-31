from Dual.dual import Dual

class AutoDiff:
    def __init__(self, fn):
        self.fn = fn

    def get_der(self, val):
        if (isinstance(val,list)):
            ders = []
            for v in val:
                x = Dual(v)
                ders.append(self.fn(x).der)
            return ders
        else:
            x = Dual(val)
            return self.fn(x).der
