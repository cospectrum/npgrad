from dataclasses import dataclass
from typing import Generic, Hashable, TypeVar

from .op import Op


NodeId = Hashable
T = TypeVar('T')


@dataclass
class Node(Generic[T]):
    id: NodeId
    op: Op[T]

    def eval(self, inputs: list[T]) -> T:
        return self.op(inputs)
