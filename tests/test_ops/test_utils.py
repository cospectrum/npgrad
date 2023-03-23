from npgrad.ops.utils import Op, NodeOp


def test_op() -> None:
    Args = list[int]

    def first(args: Args) -> int:
        return args[0]

    def second(args: Args) -> int:
        return args[1]

    def jump(f: Op[int]) -> Op[int]:

        def closure(args: Args) -> int:
            return args[f(args)]

        return closure

    inputs = [1, 0, 2]
    op = jump(first)
    assert op(inputs) == 0

    op = jump(second)
    assert op(inputs) == 1


def test_node_op() -> None:
    Args = list[float]

    def add(args: Args) -> float:
        assert len(args) == 2
        return args[0] + args[1]

    inputs = [2., 1.]
    add_node: NodeOp[float] = NodeOp(op=add)
    assert add_node.eval(inputs) == add(inputs) == add_node.op(inputs)
