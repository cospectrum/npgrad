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
