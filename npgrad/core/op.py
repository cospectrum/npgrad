from typing import Callable, TypeVar


T = TypeVar('T')
Op = Callable[[list[T]], T]
