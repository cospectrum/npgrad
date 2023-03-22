from npgrad.core import NodeOp


def test_node_op() -> None:
    Args = list[float]

    def add(args: Args) -> float:
        assert len(args) == 2
        return args[0] + args[1]

    inputs = [2., 1.]
    add_node = NodeOp(op=add)
    assert add_node.eval(inputs) == add(inputs) == add_node.op(inputs)
