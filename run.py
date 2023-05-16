import mindspore
from mindspore import nn, ops
from mindspore import numpy as mnp
from mindspore.communication import init


mindspore.set_context(mode=mindspore.GRAPH_MODE, device_target="GPU")
init(backend_name="nccl")

from model import Test

x = mnp.ones((2, 4))
print(x)

test = Test()
output = test(x)
print(output)
print(output.shape)
