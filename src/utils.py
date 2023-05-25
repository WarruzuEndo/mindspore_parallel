from typing import Tuple

import mindspore
from mindspore import ops


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


class VocabUtility:
    """Split the vocabulary into `world_size` chunks amd return the
    first and last index of the vocabulary belonging to the `rank`
    partition: Note that indices in [first, last)"""

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(
        per_partition_vocab_size: int, rank_id: int, rank_size: int
    ) -> Tuple[int, int]:
        index_f = rank_id * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(global_vocab_size: int, rank_id: int, rank_size: int) -> Tuple[int, int]:
        per_partition_vocab_size = divide_and_check_no_remainder(global_vocab_size, rank_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank_id, rank_size)
