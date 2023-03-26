from dataclasses import dataclass, field
import random
from typing import TypeVar

from npgrad.core.backprop import Function, TrainableGraph


def main() -> None:
    test_backprop()
    test_backprop_2()


Token = TypeVar('Token')


@dataclass
class F(Function[Token]):
    ctx: list[list[Token]] = field(default_factory=list)


T = float
Args = list[T]


class Add(F):
    def forward(self, args: Args) -> T:
        x, y = args
        return x + y

    def backward(self, grad: T) -> Args:
        return [grad, grad]


class Sub(F):
    def forward(self, args: Args) -> T:
        x, y = args
        return x - y

    def backward(self, grad: T) -> Args:
        return [grad, -grad]


class Mul(F):
    def forward(self, args: Args) -> T:
        x, y = args
        self.ctx.append([x, y])
        return x * y

    def backward(self, grad: T) -> Args:
        x, y = self.ctx.pop()
        return [y * grad, x * grad]


class Div(F):
    def forward(self, args: Args) -> T:
        x, y = args
        self.ctx.append([x, y])
        return x / y

    def backward(self, grad: T) -> Args:
        x, y = self.ctx.pop()
        grad_x = grad / y
        grad_y = -(x / y**2) * grad
        return [grad_x, grad_y]


def test_backprop() -> None:
    x = random.random()
    y = random.random()
    z = random.random()
    inputs = dict(x=x, y=y, z=z)

    w1 = random.random()
    w2 = random.random()
    w3 = random.random()
    weights = dict(w1=w1, w2=w2, w3=w3)

    a = w1 * x + y
    b = w2 / z
    c = a - w3
    d = c * b
    e = b - z
    sym_table = dict(a=a, b=b, c=c, d=d, e=e, w1x=w1 * x)
    sym_table.update(inputs)
    sym_table.update(weights)

    n0 = Mul(inputs=['w1', 'x'], outputs=['w1x'])
    n1 = Add(inputs=['w1x', 'y'], outputs=['a'])
    n2 = Div(inputs=['w2', 'z'], outputs=['b'])
    n3 = Sub(inputs=['a', 'w3'], outputs=['c'])
    n4 = Mul(inputs=['c', 'b'], outputs=['d'])
    n5 = Sub(inputs=['b', 'z'], outputs=['e'])

    nodes = [n0, n1, n2, n3, n4, n5]
    random.shuffle(nodes)

    graph = TrainableGraph(nodes=nodes, weights=weights)
    forward = graph.forward(inputs)
    assert forward == sym_table

    grad_d = random.random()
    grad_e = random.random()
    grads = dict(d=grad_d, e=grad_e)

    g = graph.backward(grads)
    assert g['d'] == grad_d
    assert g['e'] == grad_e

    assert g['c'] == b * g['d']
    assert g['b'] == g['e'] + c * g['d']
    assert g['z'] == -g['e'] - (w2 / z**2) * g['b']
    assert g['w2'] == g['b'] / z

    assert g['w3'] == -g['c']
    assert g['a'] == g['c']
    assert g['y'] == g['a']
    assert g['w1x'] == g['a']
    assert g['w1'] == x * g['w1x']
    assert g['x'] == w1 * g['w1x']


def test_backprop_2() -> None:
    x = random.random()
    y = random.random()
    inputs = dict(x=x, y=y)

    w = random.random()
    weights = dict(w=w)

    a = x + y
    b = a * w
    c = a - b
    d = a * c

    n0 = Add(['x', 'y'], ['a1', 'a2', 'a3'])
    n1 = Mul(['a1', 'w'], ['b'])
    n2 = Sub(['a2', 'b'], ['c'])
    n3 = Mul(['a3', 'c'], ['d'])
    nodes = [n0, n1, n2, n3]
    random.shuffle(nodes)

    graph = TrainableGraph(nodes=nodes, weights=weights)
    forward = graph.forward(inputs)

    assert forward['x'] == x
    assert forward['y'] == y
    assert forward['a1'] == forward['a2'] == forward['a3'] == a
    assert forward['b'] == b
    assert forward['c'] == c
    assert forward['d'] == d

    grad_d = random.random()
    grads = dict(d=grad_d)
    g = graph.backward(grads)
    assert g['d'] == grad_d
    assert g['c'] == a * g['d']
    assert g['b'] == -g['c']

    assert g['a1'] == w * g['b']
    assert g['a2'] == g['c']
    assert g['a3'] == c * g['d']

    assert g['w'] == a * g['b']
    assert g['x'] == g['y'] == g['a1'] + g['a2'] + g['a3']


if __name__ == '__main__':
    main()
