import mindspore
from mindspore import nn
from mindspore import ops
from mindspore import numpy as mnp
from mindspore.communication import init

mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target="GPU")
init(backend_name="nccl")

from src.layers import RowParallelLinear


def test_forward():
    linear = RowParallelLinear(
        in_features=4,
        out_features=8,
        bias=False,
        init_method=lambda x: x,
    )

    x = mnp.ones((2, 4))
    output = linear(x)

    print(output)
    print(output.shape)


def test_backward():
    linear = RowParallelLinear(
        in_features=4,
        out_features=8,
        bias=False,
        init_method=lambda x: x,
    )

    x = mnp.ones((2, 4))
    label= mnp.ones((2, 8))
    def forward(input_ids, labels):
        output = linear(input_ids)
        loss = ops.cross_entropy(output.view(-1), labels.view(-1))
        return loss
    
    grad_fn = mindspore.value_and_grad(forward, None, linear.trainable_params())
    
    def train_step(input_ids, labels):
        loss, grads = grad_fn(input_ids, labels)
        return loss

    loss = train_step(x, label)
    print(loss)


def test_value():
    linear = RowParallelLinear(
        in_features=4,
        out_features=8,
        bias=False,
        init_method=lambda x: x,
    )

    weight = mnp.ones((2, 8))
    for _, param in linear.parameters_and_names():
        param.set_data(weight)

    x = mnp.ones((2, 4))
    output = linear(x)
    print(output)


if __name__ == "__main__":
    test_forward()
    test_backward()
    test_value()