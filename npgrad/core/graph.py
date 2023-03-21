from __future__ import annotations


from dataclasses import dataclass
from typing import Generic, TypeVar

from .node import Node, NodeId, T as Token


Nodes = list[Node[Token]]
Edges = dict[NodeId, Nodes[Token]]
Results = dict[NodeId, Token]

T = TypeVar('T')


@dataclass
class Graph(Generic[T]):
    nodes: Nodes[T]
    edges: Edges[T]

    @staticmethod
    def new() -> Graph[T]:
        return Graph(nodes=[], edges=dict())

    def run(self) -> Results[T]:
        order = self.topological_order()
        results: Results[T] = dict()

        for node in order:
            inputs = [results[child.id] for child in self.children(node)]
            output = node.eval(inputs)
            results[node.id] = output

        return results

    def add_edge(self, src: Node[T], dst: Node[T]) -> None:
        edges = self.edges
        if dst.id not in edges:
            edges[dst.id] = []
        edges[dst.id].append(src)

    def add_node(self, node: Node[T]) -> None:
        self.nodes.append(node)

    def children(self, node: Node[T]) -> Nodes[T]:
        return self.edges.get(node.id, [])

    def topological_order(self) -> Nodes[T]:
        order = []
        visited = set()

        def visit(node: Node[T]) -> None:
            if node.id in visited:
                return
            visited.add(node.id)
            children = self.children(node)
            for child in children:
                visit(child)
            order.append(node)

        for node in self.nodes:
            visit(node)
        return order
