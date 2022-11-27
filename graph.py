from graphviz import Digraph

def draw(node):
    graph = Digraph()
    graph.attr(rankdir='LR', size='8,5')
    def _draw(node):
        # Put node into graph   
        node_id = f'{id(node)}'
        label = f'{node.label} | {node.data}' if node.label else f'{node.data}'
        graph.node(node_id, label)
        # Return if it's leaf node
        if node._prev == None:
            return
        # For binary operations
        if len(node._prev) == 2:
            # Get child nodes
            (x, y) = node._prev
            (x_id, y_id) = (f'{id(x)}', f'{id(y)}')
            # Create operation node and put into graph
            op_id = f'{node_id}{node._op}{x_id}{y_id}'
            graph.node(op_id, node._op)
            # Connect opeartion node to the main node
            graph.edge(op_id, node_id)
            # Connect child nodes to the operation
            graph.edge(x_id, op_id)
            graph.edge(y_id, op_id)
            _draw(x)
            _draw(y)
        if len(node._prev) == 1:
            # Get child node
            x = node._prev[0]
            x_id = f'{id(x)}'
            # Create operation node and put into graph
            op_id = f'{node_id}{node._op}{x_id}'
            graph.node(op_id, node._op)
            # Connect opeartion node to the main node
            graph.edge(op_id, node_id)
            # Connect child node to the operation
            graph.edge(x_id, op_id)
            _draw(x)
    _draw(node)
    return graph
