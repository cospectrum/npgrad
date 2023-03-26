from typing import Callable, Iterable, Sequence, TypeVar
from .graph import Graph, TokenMap
from .node import Node, T as Token


Grads = TokenMap[Token]
SumFn = Callable[[Iterable[Token]], Token]

T = TypeVar('T')


class Function(Node[T]):

    def forward(self, args: list[T]) -> T:
        raise NotImplementedError

    def backward(self, grad: T) -> list[T]:
        raise NotImplementedError


class TrainableGraph(Graph[T]):
    nodes: Sequence[Function[T]]
    sum_fn: SumFn[T]

    def __init__(
        self,
        nodes: Sequence[Function[T]],
        weights: TokenMap[T] | None = None,
        sum_fn: SumFn[T] = sum,  # type: ignore
    ) -> None:
        super().__init__(nodes, weights)
        self.sum_fn = sum_fn

    def forward(self, inputs: TokenMap[T]) -> TokenMap[T]:
        sym_table = self.weights.copy()
        sym_table.update(inputs)

        for node in self.nodes:
            args = [sym_table[token_id] for token_id in node.inputs]
            out = node.forward(args)
            for token_id in node.outputs:
                sym_table[token_id] = out
        return sym_table

    def backward(self, grads: Grads[T]) -> Grads[T]:
        grads = grads.copy()

        def add(left: T | None, right: T) -> T:
            if left is None:
                return right
            return self.sum_fn((left, right))

        for node in reversed(self.nodes):
            grad = self.sum_fn(grads[token_id] for token_id in node.outputs)
            jvps = node.backward(grad)
            for token_id, jvp in zip(node.inputs, jvps):
                grads[token_id] = add(grads.get(token_id), jvp)
        return grads
