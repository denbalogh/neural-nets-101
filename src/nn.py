from src.value import Value
from random import uniform

class Neuron:
    def __init__(self, nin, activation='sigmoid'):
        self.activation = activation
        # Random initialization of weights and bias
        self.weights = [Value(uniform(-1, 1)) for _ in range(nin)]
        self.bias = Value(0)

    # Forward pass
    def __call__(self, values):
        val = sum([w * v for w, v in zip(self.weights, values)], self.bias)
        if self.activation == 'sigmoid':
            return val.sigmoid()
        if self.activation == 'tanh':
            return val.tanh()
