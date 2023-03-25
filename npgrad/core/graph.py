from typing import Generic, Sequence, TypeVar

from .node import Node, TokenId, T as Token
from .order import Order, topological_order

from ..utils import immutable


Nodes = Sequence[Node[Token]]
TokenMap = dict[TokenId, Token]

T = TypeVar('T')


@immutable('nodes', 'order')
class Graph(Generic[T]):
    nodes: Nodes[T]
    order: Order

    def __init__(self, nodes: Nodes[T]) -> None:
        self.nodes = nodes
        self.order = topological_order(nodes)

    def eval(self, inputs: TokenMap[T]) -> TokenMap[T]:
        sym_table = inputs.copy()
        ordered_nodes = (self.nodes[index] for index in self.order)

        for node in ordered_nodes:
            args = [sym_table[token_id] for token_id in node.inputs]
            out = node.eval(args)
            for token_id in node.outputs:
                sym_table[token_id] = out
        return sym_table


class WeightedGraph(Graph[T]):
    weights: TokenMap[T]

    def __init__(
        self,
        nodes: Nodes[T],
        weights: TokenMap[T] | None = None
    ) -> None:
        super().__init__(nodes)
        if weights is None:
            weights = dict()
        self.weights = weights

    def eval(self, inputs: TokenMap[T]) -> TokenMap[T]:
        _inputs = self.weights.copy()
        for k in inputs:
            _inputs[k] = inputs[k]
        return super().eval(_inputs)
