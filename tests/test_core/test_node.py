from npgrad.core import Node


def test_node() -> None:

    def add(args: list[float]) -> float:
        assert len(args) == 2
        return args[0] + args[1]

    inputs = [2., 1.]
    add_node = Node(op=add, id=0)
    assert add_node.eval(inputs) == add(inputs) == add_node.op(inputs)
