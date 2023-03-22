from __future__ import annotations

from typing import Any, Generic, TypeVar

from .op import Op


TokenId = Any
TokenIds = list[TokenId]
T = TypeVar('T')


class Node(Generic[T]):
    inputs: TokenIds
    outputs: TokenIds

    def __init__(
        self,
        inputs: TokenIds | None = None,
        outputs: TokenIds | None = None,
    ) -> None:
        if inputs is None:
            inputs = []
        if outputs is None:
            outputs = []
        self.inputs = inputs
        self.outputs = outputs

    def eval(self, args: list[T]) -> T:
        raise TypeError

    def __hash__(self) -> int:
        return id(self)


class NodeOp(Node[T]):
    op: Op[T]

    def __init__(
        self,
        op: Op[T],
        inputs: TokenIds | None = None,
        outputs: TokenIds | None = None,
    ) -> None:
        super().__init__(inputs=inputs, outputs=outputs)
        self.op = op

    def eval(self, args: list[T]) -> T:
        return self.op(args)
