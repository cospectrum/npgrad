import random

from npgrad.core import Graph, Node


def test_graph() -> None:
    Args = list[float]

    class Add(Node[float]):
        def eval(self, args: Args) -> float:
            assert len(args) == 2
            return args[0] + args[1]

    class Mul(Node[float]):
        def eval(self, args: Args) -> float:
            assert len(args) == 2
            return args[0] * args[1]

    class Sub(Node[float]):
        def eval(self, args: Args) -> float:
            assert len(args) == 2
            return args[0] - args[1]

    class Div(Node):
        def eval(self, args: Args) -> float:
            assert len(args) == 2
            return args[0] / args[1]

    x = random.random()
    y = random.random()
    z = random.random()

    a = x + y
    b = y / z
    c = b - x
    d = c * a

    a_node = Add(inputs=['x', 'y'], outputs=['a'])
    b_node = Div(inputs=['y', 'z'], outputs=['b'])
    c_node = Sub(inputs=['b', 'x'], outputs=['c'])
    d_node = Mul(inputs=['c', 'a'], outputs=['d'])

    nodes = [a_node, b_node, c_node, d_node]
    random.shuffle(nodes)
    g = Graph(nodes=nodes)

    inputs = {'x': x, 'y': y, 'z': z}
    sym_table = g.run(inputs)

    assert sym_table['x'] == x
    assert sym_table['y'] == y
    assert sym_table['z'] == z
    assert sym_table['a'] == a
    assert sym_table['b'] == b
    assert sym_table['c'] == c
    assert sym_table['d'] == d
