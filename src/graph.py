from graphviz import Digraph

def get_graph(node):
    graph = Digraph(strict=True)
    graph.attr(rankdir='LR')
    def _draw(node):
        # Put node into graph   
        node_id = f'{id(node)}'
        label = f'<f0> {node.label}|<f1> {node.data}|<f2> {node.grad}' if node.label else f'<f1> {node.data}|<f2> {node.grad}'
        graph.node(node_id, label, shape='record', rankdir='LR')
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
            graph.edge(op_id, f'{node_id}:f1')
            # Connect child nodes to the operation
            graph.edge(f'{x_id}:f1', op_id)
            graph.edge(f'{y_id}:f1', op_id)
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
            graph.edge(op_id, f'{node_id}:f1')
            # Connect child node to the operation
            graph.edge(f'{x_id}:f1', op_id)
            _draw(x)
    _draw(node)
    return graph
