from typing import Any, Callable


def immutable(*immutable_attrs: str) -> Callable:
    '''Makes the given attributes immutable for the instance.
    By default, all instance attributes will be set to immutable.
    '''

    frozen = len(immutable_attrs) == 0

    def decorator(cls):

        def setattr(self, name: str, value: Any) -> None:
            if hasattr(self, name) and (name in immutable_attrs or frozen):
                raise AttributeError(f'Attribute {name} is immutable')
            super(cls, self).__setattr__(name, value)

        cls.__setattr__ = setattr
        return cls

    return decorator
