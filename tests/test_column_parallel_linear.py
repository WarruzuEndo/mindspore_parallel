import mindspore
from mindspore import nn
from mindspore import ops
from mindspore import numpy as mnp
from mindspore.communication import init

mindspore.set_context(mode=mindspore.GRAPH_MODE, device_target="GPU")
init(backend_name="nccl")

from src.layers import ColumnParallelLinear


def test_forward():
    linear = ColumnParallelLinear(
        in_features=4,
        out_features=8,
        bias=False,
        gather_output=True,
        init_method=lambda x: x,
    )

    x = mnp.ones((2, 4))
    output = linear(x)

    print(output)
    print(output.shape)


if __name__ == "__main__":
    test_forward()
