from graphviz import Digraph

def get_graph(node):
    graph = Digraph(strict=True)
    graph.attr(rankdir='LR')
    def _draw(node):
        node_id = str(id(node))
        label = f'<f0> {node.label}|<f1> {node.data:.4f}|<f2> {node.grad:.4f}' if node.label else f'<f1> {node.data:.4f}|<f2> {node.grad:.4f}'
        graph.node(node_id, label, shape='record', rankdir='LR')
        
        if node._prev == None:
            return
        
        # For binary operations
        if len(node._prev) == 2:
            (x, y) = node._prev
            (x_id, y_id) = (str(id(x)), str(id(y)))
            op_id = f'{node_id}{node._op}{x_id}{y_id}'
            graph.node(op_id, node._op)
            graph.edge(op_id, f'{node_id}:f1')
            graph.edge(f'{x_id}:f1', op_id)
            graph.edge(f'{y_id}:f1', op_id)
            _draw(x)
            _draw(y)
            
        # For unary operations
        if len(node._prev) == 1:
            x = node._prev[0]
            x_id = str(id(x))
            op_id = f'{node_id}{node._op}{x_id}'
            graph.node(op_id, node._op)
            graph.edge(op_id, f'{node_id}:f1')
            graph.edge(f'{x_id}:f1', op_id)
            _draw(x)
            
    _draw(node)
    return graph
