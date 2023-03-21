from npgrad.core import Op


def test_op() -> None:

    def first(args: list[int]) -> int:
        return args[0]

    def second(args: list[int]) -> int:
        return args[1]

    def jump(f: Op[int]) -> Op[int]:

        def closure(args: list[int]) -> int:
            return args[f(args)]

        return closure

    inputs = [1, 0, 2]
    op = jump(first)
    assert op(inputs) == 0

    op = jump(second)
    assert op(inputs) == 1
