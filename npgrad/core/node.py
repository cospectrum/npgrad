from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from ..utils import immutable


TokenId = Any
TokenIds = list[TokenId]

T = TypeVar('T')


@dataclass
@immutable('inputs', 'outputs')
class Node(Generic[T]):
    inputs: TokenIds = field(default_factory=list)
    outputs: TokenIds = field(default_factory=list)

    def eval(self, args: list[T]) -> T:
        raise NotImplementedError
