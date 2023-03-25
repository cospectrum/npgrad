from typing import Sequence

from .node import Node, TokenIds


__all__ = ['NodeIndex', 'Order', 'topological_order']

NodeIndex = int
Nodes = Sequence[Node]
Order = list[NodeIndex]

Pair = tuple[NodeIndex, Node]


def topological_order(nodes: Nodes) -> Order:
    order = []
    visited = set()

    def visit(node: Node, index: NodeIndex) -> None:
        if index in visited:
            return
        visited.add(index)
        for child_index, child in children(nodes, node):
            visit(child, child_index)
        order.append(index)

    for index, node in enumerate(nodes):
        visit(node, index)
    return order


def children(nodes: Nodes, parent: Node) -> list[Pair]:
    inputs = set(parent.inputs)

    def wired(ids: TokenIds) -> bool:
        return not inputs.isdisjoint(ids)

    return [
        (index, node) for index, node in enumerate(nodes)
        if wired(node.outputs)
    ]
