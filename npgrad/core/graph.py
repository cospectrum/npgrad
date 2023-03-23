from dataclasses import dataclass
from typing import Generic, Sequence, TypeVar

from .node import Node, TokenId, TokenIds, T as Token


Nodes = Sequence[Node[Token]]
TokenMap = dict[TokenId, Token]
NodeIndex = int
Order = list[NodeIndex]

T = TypeVar('T')


@dataclass
class Graph(Generic[T]):
    nodes: Nodes[T]

    def run(self, inputs: TokenMap[T]) -> TokenMap[T]:
        order = self.topological_order()
        return self.run_with_order(inputs, order)

    def run_with_order(self, inputs: TokenMap, order: Order) -> TokenMap[T]:
        sym_table = inputs.copy()
        ordered_nodes = (self.nodes[index] for index in order)

        for node in ordered_nodes:
            args = [sym_table[token_id] for token_id in node.inputs]
            out = node.eval(args)
            for token_id in node.outputs:
                sym_table[token_id] = out
        return sym_table

    def topological_order(self) -> Order:
        order = []
        visited = set()

        def visit(node: Node[T], index: NodeIndex) -> None:
            if index in visited:
                return
            visited.add(index)
            for child_index in self.children_indexes(node):
                child = self.nodes[child_index]
                visit(child, child_index)
            order.append(index)

        for index, node in enumerate(self.nodes):
            visit(node, index)
        return order

    def children_indexes(self, node: Node[T]) -> list[NodeIndex]:
        inputs = node.inputs

        def wired(ids: TokenIds) -> bool:
            return len(set(ids).intersection(inputs)) != 0

        return [
            index for index, other_node in enumerate(self.nodes)
            if wired(other_node.outputs)
        ]
