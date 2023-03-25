import random
import pytest

from npgrad.core import Graph, Node


@pytest.mark.xfail(raises=AttributeError)
def test_graph_nodes_mutability() -> None:
    node: Node = Node()
    nodes = [node]

    graph = Graph(nodes=nodes)
    assert graph.weights == dict()
    assert graph.nodes != []
    graph.nodes = []


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

    w1 = random.random()
    w2 = random.random()
    weights = {'w1': w1, 'w2': w2}

    w1x = w1 * x
    a = w1x + y
    b = y / z
    w2_div_b = w2 / b
    c = w2_div_b - x
    d = c * a

    n0 = Mul(inputs=['w1', 'x'], outputs=['w1*x'])
    n1 = Add(inputs=['w1*x', 'y'], outputs=['a'])
    n2 = Div(inputs=['y', 'z'], outputs=['b'])
    n3 = Div(inputs=['w2', 'b'], outputs=['w2/b'])
    n4 = Sub(inputs=['w2/b', 'x'], outputs=['c'])
    n5 = Mul(inputs=['c', 'a'], outputs=['d'])

    nodes = [n0, n1, n2, n3, n4, n5]
    random.shuffle(nodes)
    g = Graph(nodes=nodes, weights=weights)

    inputs = {'x': x, 'y': y, 'z': z}
    sym_table = g.eval(inputs)

    assert sym_table['x'] == x
    assert sym_table['y'] == y
    assert sym_table['z'] == z
    assert sym_table['a'] == a
    assert sym_table['b'] == b
    assert sym_table['c'] == c
    assert sym_table['d'] == d

    assert sym_table['w1'] == w1
    assert sym_table['w2'] == w2
    assert sym_table['w1*x'] == w1x
    assert sym_table['w2/b'] == w2_div_b
