import random
import pytest

from npgrad.core import Graph, Node


@pytest.mark.xfail(raises=AttributeError)
def test_graph_nodes_mutability() -> None:
    node: Node = Node()
    nodes = [node]

    graph = Graph(nodes=nodes)
    assert graph.nodes != []
    graph.nodes = []


@pytest.mark.xfail(raises=AttributeError)
def test_graph_order_mutability() -> None:
    node: Node[float] = Node()
    nodes = [node]

    graph = Graph(nodes=nodes)
    assert graph.order != []
    graph.order = []


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

    n1 = Add(inputs=['x', 'y'], outputs=['a'])
    n2 = Div(inputs=['y', 'z'], outputs=['b'])
    n3 = Sub(inputs=['b', 'x'], outputs=['c'])
    n4 = Mul(inputs=['c', 'a'], outputs=['d'])

    nodes = [n1, n2, n3, n4]
    random.shuffle(nodes)
    g = Graph(nodes=nodes)

    inputs = {'x': x, 'y': y, 'z': z}
    sym_table = g.eval(inputs)

    assert sym_table['x'] == x
    assert sym_table['y'] == y
    assert sym_table['z'] == z
    assert sym_table['a'] == a
    assert sym_table['b'] == b
    assert sym_table['c'] == c
    assert sym_table['d'] == d
