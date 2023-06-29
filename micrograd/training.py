def backpropagate(node):
    node.grad = 1
    def _backpropagate(node):
        if node._prev == None:
            return
        node._backward()
        for prev in node._prev:
            _backpropagate(prev)
    _backpropagate(node)

def train(model, x, y_label, lr=0.01, loss_func='mns'):
    for p in model.parameters():
        p.grad = 0
        
    y = model.forward(x)
    
    loss = 0
    if loss_func == 'mns':
        loss = mean_squared_error(y, y_label)
    elif loss_func == 'ce':
        loss = cross_entropy(y, y_label)
        
    backpropagate(loss)
    
    for p in model.parameters():
        p.data -= lr * p.grad
    return loss

# Loss functions

def mean_squared_error(y, y_label):
    return sum([(y_label - y)**2 for y_label, y in zip(y_label, y)]) / len(y)

def cross_entropy(y, y_label):
    loss = -sum([y_label * y.log() for y_label, y in zip(y_label, y)])
    return loss / len(y)
