class Value:
    def __init__(self, data, label='', _op='', _prev=None):
        self.label = label
        self.data = data
        self._op = _op
        self._prev = _prev
        
    
    def __repr__(self):
        return f'Value(data={self.data})'
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _op='+', _prev=(self, other))
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _op='*', _prev=(self, other))
        return out
    
    def __pow__(self, exponent):
        out = Value(self.data ** exponent, _op=f'** {exponent}', _prev=(self,))
        return out
