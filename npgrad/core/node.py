from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar


TokenId = Any
TokenIds = list[TokenId]

T = TypeVar('T')


@dataclass
class Node(Generic[T]):
    inputs: TokenIds = field(default_factory=list)
    outputs: TokenIds = field(default_factory=list)

    def eval(self, args: list[T]) -> T:
        raise TypeError('unimplemented')
