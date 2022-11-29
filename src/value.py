from math import exp
class Value:
    def __init__(self, data, label='', _op='', _prev=None):
        self.data = data
        # Graph variables
        self.label = label
        self._op = _op
        self._prev = _prev
        # Backpropagation variables
        self.grad = 0
        self._backward = lambda: None
    
    def __repr__(self):
        return f'Value(data={self.data:.4f},label={self.label})'
    
    # Unary operations
    
    def __neg__(self):
        out = Value(-1 * self.data, _op='neg', _prev=(self,))
        
        def _backward():
            self.grad += out.grad * -1
        out._backward = _backward
        
        return out
    
    # Binary operations
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _op='+', _prev=(self, other))
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        
        return out
    
    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _op='*', _prev=(self, other))
        
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        
        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other**(-1)
    
    def __pow__(self, exponent):
        out = Value(self.data ** exponent, _op=f'exp({exponent})', _prev=(self,))
        
        def _backward():
            self.grad += out.grad * exponent * self.data ** (exponent - 1)
        out._backward = _backward
        
        return out

    # Reverse binary operations
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __rtruediv__(self, other):
        return Value(other) / self
    
    def __rsub__(self, other):
        return Value(other) - self

    # Activation functions
    def sigmoid(self):
        data = 1 / (1 + exp(-self.data))
        out = Value(data, _op='sigmoid', _prev=(self,))
        
        def _backward():
            self.grad += out.grad * out.data * (1 - out.data)
        out._backward = _backward
        
        return out
    
    def tanh(self):
        data = (exp(self.data) - exp(-self.data)) / (exp(self.data) + exp(-self.data))
        out = Value(data, _op='tanh', _prev=(self,))
        
        def _backward():
            self.grad += out.grad * (1 - out.data ** 2)
        out._backward = _backward
        
        return out
