import pytest
from npgrad.core import Node


@pytest.mark.xfail(raises=AttributeError)
def test_node_mutability() -> None:
    node: Node = Node(inputs=['x'])
    assert node.inputs != []
    assert node.outputs == []

    node.inputs = []


def test_node_init() -> None:
    inputs = ['x']
    node: Node = Node(['x'])
    assert node.inputs == inputs
    assert len(node.outputs) == 0
