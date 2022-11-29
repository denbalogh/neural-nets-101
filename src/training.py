def backpropagate(node):
    # Initialize last node gradient to 1
    node.grad = 1
    # Calculate gradients for all nodes
    def _backpropagate(node):
        if node._prev == None:
            return
        node._backward()
        for prev in node._prev:
            _backpropagate(prev)
    _backpropagate(node)

def train(model, x, y_label, lr=0.01):
    # Set gradients to 0
    for p in model.parameters():
        p.grad = 0
    # Forward pass
    y = model.forward(x)
    # Calculate loss
    loss = mean_squared_error(y, y_label)
    # Backward pass
    backpropagate(loss)
    # Update weights
    for p in model.parameters():
        p.data -= lr * p.grad
    return loss

# Loss functions

def mean_squared_error(y, y_label):
    return sum([(y_label - y)**2 for y_label, y in zip(y_label, y)]) / len(y)
