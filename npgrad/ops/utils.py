from dataclasses import dataclass
from typing import Callable, TypeVar

from ..core import Node


T = TypeVar('T')
Op = Callable[[list[T]], T]


@dataclass(kw_only=True)
class NodeOp(Node[T]):
    op: Op[T]

    def eval(self, args: list[T]) -> T:
        return self.op(args)
