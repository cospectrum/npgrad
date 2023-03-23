from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Sequence, TypeVar

from .node import Node, TokenId, TokenIds, T as Token


Nodes = Sequence[Node[Token]]
TokenMap = dict[TokenId, Token]

T = TypeVar('T')


@dataclass
class Graph(Generic[T]):
    nodes: Nodes[T]

    def run(self, inputs: TokenMap[T]) -> TokenMap[T]:
        order = self.topological_order()
        return self.run_with_order(inputs, order)

    def run_with_order(self, inputs: TokenMap, order: Nodes[T]) -> TokenMap[T]:
        sym_table = inputs.copy()

        for node in order:
            args = [sym_table[token_id] for token_id in node.inputs]
            out = node.eval(args)
            for token_id in node.outputs:
                sym_table[token_id] = out
        return sym_table

    def topological_order(self) -> Nodes[T]:
        order = []
        visited = set()

        def visit(node: Node[T]) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            children = self.children(node)
            for child in children:
                visit(child)
            order.append(node)

        for node in self.nodes:
            visit(node)
        return order

    def children(self, node: Node[T]) -> Nodes[T]:
        inputs = node.inputs

        def wired(ids: TokenIds) -> bool:
            return len(set(ids).intersection(inputs)) != 0

        return [n for n in self.nodes if wired(n.outputs)]
