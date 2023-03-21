import random

from npgrad.core import Graph, Node


def test_graph() -> None:
    Args = list[float]

    def add(args: Args) -> float:
        assert len(args) == 2
        return args[0] + args[1]

    def mul(args: Args) -> float:
        assert len(args) == 2
        return args[0] * args[1]

    def sub(args: Args) -> float:
        assert len(args) == 2
        return args[0] - args[1]

    def div(args: Args) -> float:
        assert len(args) == 2
        return args[0] / args[1]

    x = random.random()
    y = random.random()
    z = random.random()

    a = x + y
    b = y / z
    c = b - x
    d = c * a

    x_node = Node(op=lambda _: x, id='x')
    y_node = Node(op=lambda _: y, id='y')
    z_node = Node(op=lambda _: z, id='z')

    a_node = Node(op=add, id='a')
    b_node = Node(op=div, id='b')
    c_node = Node(op=sub, id='c')
    d_node = Node(op=mul, id='d')

    g: Graph = Graph.new()
    nodes = [x_node, y_node, z_node, a_node, b_node, c_node, d_node]
    random.shuffle(nodes)
    for node in nodes:
        g.add_node(node)

    # a
    g.add_edge(x_node, a_node)
    g.add_edge(y_node, a_node)

    # b
    g.add_edge(y_node, b_node)
    g.add_edge(z_node, b_node)

    # c
    g.add_edge(b_node, c_node)
    g.add_edge(x_node, c_node)

    # d
    g.add_edge(c_node, d_node)
    g.add_edge(a_node, d_node)

    order = [node.id for node in g.topological_order()]
    print(f'{order=}')

    results = g.run()
    assert results[d_node.id] == d
