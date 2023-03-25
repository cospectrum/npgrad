from typing import Generic, Sequence, TypeVar

from .node import Node, TokenId, T as Token
from .order import topological_order

from ..utils import immutable


Nodes = Sequence[Node[Token]]
TokenMap = dict[TokenId, Token]

T = TypeVar('T')


@immutable('nodes')
class Graph(Generic[T]):
    weights: TokenMap[T]
    nodes: Nodes[T]

    def __init__(
        self,
        nodes: Nodes[T],
        weights: TokenMap[T] | None = None,
    ) -> None:
        order = topological_order(nodes)
        self.nodes = [nodes[index] for index in order]
        if weights is None:
            weights = dict()
        self.weights = weights

    def eval(self, inputs: TokenMap[T]) -> TokenMap[T]:
        sym_table = self.weights.copy()
        sym_table.update(inputs)

        for node in self.nodes:
            args = [sym_table[token_id] for token_id in node.inputs]
            out = node.eval(args)
            for token_id in node.outputs:
                sym_table[token_id] = out
        return sym_table
