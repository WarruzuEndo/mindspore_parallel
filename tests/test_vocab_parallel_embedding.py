import mindspore
from mindspore import nn
from mindspore import ops
from mindspore import numpy as mnp
from mindspore.communication import init

mindspore.set_context(mode=mindspore.GRAPH_MODE, device_target="GPU")
init(backend_name="nccl")

from src.layers import VocabParallelEmbedding


def test_forward():
    embedding = VocabParallelEmbedding(
        vocab_size=20000,
        embedding_size=768
    )

    x = mnp.ones((8, 128), dtype=mindspore.int32)
    output = embedding(x)

    print(output.shape)


def test_backward():
    embedding = VocabParallelEmbedding(
        vocab_size=20000,
        embedding_size=768
    )

    x = mnp.ones((8, 128), dtype=mindspore.int32)
    label= mnp.ones((8, 128, 768))
    def forward(input_ids, labels):
        output = embedding(input_ids)
        loss = ops.cross_entropy(output.view(-1), labels.view(-1))
        return loss

    grad_fn = mindspore.value_and_grad(forward, None, embedding.trainable_params())

    def train_step(input_ids, labels):
        loss, grads = grad_fn(input_ids, labels)
        return loss

    loss = train_step(x, label)
    print(loss)


if __name__ == "__main__":
    test_forward()
    test_backward()
