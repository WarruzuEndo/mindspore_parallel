import math
from typing import Any, Callable, Tuple, Optional
import numpy as np

import mindspore
from mindspore import nn
from mindspore import ops
from mindspore import numpy as mnp

from mindspore.communication import get_rank, get_group_size


def ensure_divisibility(numerator: int, denominator: int) -> None:
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)


def divide_and_check_no_remainder(numerator: int, denominator: int) -> int:
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(
    tensor: mindspore.Tensor, num_partitions: int
) -> Tuple[mindspore.Tensor, ...]:
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
    """
    # Get the size and dimension.
    last_dim = tensor.ndim - 1
    # Split.
    last_dim_size = divide_and_check_no_remainder(tensor.shape[last_dim], num_partitions)
    tensor_list = ops.split(tensor, last_dim_size, axis=last_dim)

    return tensor_list


class _CopyToModelParallelRegion(nn.Cell):
    """Pass the input to the model parallel region."""
    def __init__(self):
        super().__init__()
        self.rank_size = get_group_size()
        self.all_reduce = ops.AllReduce()

    def construct(self, input_):  # type: ignore
        return input_

    def bprop(self, input_, out, dout):  # type: ignore
        return self._reduce(dout)

    def _reduce(self, input_: mindspore.Tensor) -> mindspore.Tensor:
        """All-reduce the the input tensor across model parallel group."""
        # Bypass the function if we are using only 1 GPU.
        if self.rank_size == 1:
            return input_

        # All-reduce.
        self.all_reduce(input_)

        return input_


class _GatherFromModelParallelRegion(nn.Cell):
    """Gather the input from model parallel region and concatinate."""
    def __init__(self):
        super().__init__()
        self.rank_id = get_rank()
        self.rank_size = get_group_size()
        self.all_gather = ops.AllGather()

    def construct(self, input_):  # type: ignore
        return self._gather(input_)

    def bprop(self, input_, out, dout):  # type: ignore
        return self._split(dout)

    def _gather(self, input_: mindspore.Tensor) -> mindspore.Tensor:
        """Gather tensors and concatinate along the last dimension."""
        # Bypass the function if we are using only 1 GPU.
        if self.rank_size  == 1:
            return input_

        # Size and dimension.
        last_dim = input_.ndim - 1

        tensor = self.all_gather(input_)
        tensor_list = ops.split(tensor, divide_and_check_no_remainder(tensor.shape[0], self.rank_size), axis=0)
        output = ops.concat(tensor_list, axis=last_dim)

        return output
    
    def _split(self, input_: mindspore.Tensor) -> mindspore.Tensor:
        """Split the tensor along its last dimension and keep the
        corresponding slice."""
        # Bypass the function if we are using only 1 GPU.
        if  self.rank_size == 1:
            return input_

        # Split along last dimension.
        input_list = split_tensor_along_last_dim(input_, self.rank_size)

        # Note: torch.split does not create contiguous tensors by default.
        output = input_list[self.rank_id]

        return output


_copyToModel = _CopyToModelParallelRegion()
_gatherFromModel = _GatherFromModelParallelRegion()


def copy_to_model_parallel_region(input_: mindspore.Tensor) -> mindspore.Tensor:
    return _copyToModel(input_)


def gather_from_model_parallel_region(input_: mindspore.Tensor) -> mindspore.Tensor:
    return _gatherFromModel(input_)


def _initialize_affine_weight(
    weight: mindspore.Tensor,
    out_features: int,
    in_features: int,
    per_partition_size: int,
    partition_dim: int,
    init_method: Callable[[mindspore.Tensor], mindspore.Tensor],
    stride: int = 1,
    return_master_weight: bool = False,
) -> Optional[mindspore.Tensor]:
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    # If we only use 1 process for model parallelism, bypass scatter.
    rank_size = get_group_size()
    if rank_size == 1:
        init_method(weight)
        if return_master_weight:
            return weight
        return None

    # Initialize master weight
    master_weight = mnp.empty((out_features, in_features), dtype=weight.dtype)
    init_method(master_weight)

    # Split and copy
    per_partition_per_stride_size = divide_and_check_no_remainder(per_partition_size, stride)
    weight_list = ops.split(master_weight, per_partition_per_stride_size, axis=partition_dim)
    rank_id = get_rank()
    my_weight_list = weight_list[rank_id::rank_size]

    weight = ops.concat(my_weight_list, axis=partition_dim)
    ops.stop_gradient(weight)

    if return_master_weight:
        return master_weight
    return None


class ColumnParallelLinear(nn.Cell):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        in_features: first dimension of matrix A.
        out_features: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
        init_method: Callable[[mindspore.Tensor], mindspore.Tensor] = mindspore.common.initializer.XavierNormal,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
    ) -> None:
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        rank_size = get_group_size()
        self.output_size_per_partition = divide_and_check_no_remainder(out_features, rank_size)

        # Parameters.
        self.weight = mindspore.Parameter(mindspore.Tensor(np.random.randn(self.in_features, self.output_size_per_partition)))
        if bias:
            self.bias = mindspore.Parameter(mindspore.Tensor(np.random.randn(self.output_size_per_partition)))
            # Always initialize bias to zero.
            with ops.stop_gradient():
                self.bias = mnp.zeros_like(self.bias)
        else:
            self.bias = None

        # Initialize weight.
        self.master_weight = _initialize_affine_weight(
            self.weight,
            self.out_features,
            self.in_features,
            self.output_size_per_partition,
            0,
            init_method,
            stride=stride,
            return_master_weight=keep_master_weight_for_test,
        )

    def get_master_weight(self) -> mindspore.Tensor:
        return gather_from_model_parallel_region(self.weight.data.transpose(0, 1)).transpose_(0, 1)

    def construct(self, input_: mindspore.Tensor) -> mindspore.Tensor:  # type: ignore
        # Set up backprop all-reduce.
        input_parallel = copy_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = ops.matmul(input_parallel, self.weight)
        if self.bias is not None:
            output_parallel = ops.bias_add(input_parallel, self.bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        return output
