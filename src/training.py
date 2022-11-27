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
