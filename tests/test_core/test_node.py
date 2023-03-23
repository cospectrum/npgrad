from npgrad.core import Node


def test_node_init() -> None:
    inputs = ['x']
    node: Node = Node(['x'])
    assert node.inputs == inputs
    assert len(node.outputs) == 0
