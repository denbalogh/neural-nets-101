from src.value import Value
from random import uniform

class Neuron:
    def __init__(self, nin, activation=None):
        self.activation = activation
        self.weights = [Value(uniform(-1, 1), f'w{idx}') for idx in range(nin)]
        self.bias = Value(0, 'b')

    def __call__(self, values):
        val = sum([w * v for w, v in zip(self.weights, values)], self.bias)
        if self.activation == 'sigmoid':
            return val.sigmoid()
        if self.activation == 'tanh':
            return val.tanh()
        if self.activation == 'relu':
            return val.relu()
        if self.activation == 'leaky_relu':
            return val.leaky_relu()
        else:
            return val
        
    def parameters(self):
        return self.weights + [self.bias]

class Layer:
    def __init__(self, nin, nout, activation='relu'):
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]
        self.activation = activation
        
    def __call__(self, values):
        values = [n(values) for n in self.neurons]
        if self.activation == 'softmax':
            return self.softmax(values)
        
        return values
    
    def softmax(self, values):
        exps = [v.exp() for v in values]
        exps_sum = sum(exps)
        
        return [exp / exps_sum for exp in exps]
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, nin, layers, hidden_activation='relu', last_activation='relu'):
        layers = [nin] + layers
        self.layers = []
        for idx in range(len(layers) - 1):
            activation = hidden_activation if idx < len(layers) - 2 else last_activation
            self.layers.append(Layer(layers[idx], layers[idx + 1], activation))
        
    def forward(self, values):
        for layer in self.layers:
            values = layer(values)
        return values
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
