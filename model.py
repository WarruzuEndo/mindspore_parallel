import mindspore
from mindspore import nn
from layers import ColumnParallelLinear

class Test(nn.Cell):
    def __init__(self):
        super().__init__()

        self.linear = ColumnParallelLinear(
            in_features=4,
            out_features=8,
            bias=False,
            gather_output=True,
            init_method=lambda x: x,
        )

    def construct(self, x: mindspore.Tensor):
        output = self.linear(x)
        return output