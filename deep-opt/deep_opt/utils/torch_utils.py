from typing import Union, Any
import torch


def tensor_like(orig: torch.Tensor, new_data: Any) -> torch.Tensor:
    """
    Creates a new tensor using torch.tensor that has the same
    dtype, device and require_grad as the first argument.
    :param orig: The tensor whose attributes the new tensor should have
    :param new_data: The data for the new tensor. Can be any type that torch.tensor can handle.
    :return: A new Tensor with the given data and the attributes of the given tensor.
    """
    return torch.tensor(
        data=new_data,
        dtype=orig.dtype,
        device=orig.device,
        requires_grad=orig.requires_grad
    )
