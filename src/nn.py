from src.value import Value
from random import uniform

class Neuron:
    def __init__(self, nin, activation='sigmoid'):
        self.activation = activation
        # Random initialization of weights and bias
        self.weights = [Value(uniform(-1, 1), f'w{idx}') for idx in range(nin)]
        self.bias = Value(0, 'b')

    # Forward pass
    def __call__(self, values):
        val = sum([w * v for w, v in zip(self.weights, values)], self.bias)
        if self.activation == 'sigmoid':
            return val.sigmoid()
        if self.activation == 'tanh':
            return val.tanh()
        
    def parameters(self):
        return self.weights + [self.bias]

class Layer:
    def __init__(self, nin, nout, activation='sigmoid'):
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]
        
    def __call__(self, values):
        return [n(values) for n in self.neurons]
    
    def parameters(self):
        return [n.parameters() for n in self.neurons]
