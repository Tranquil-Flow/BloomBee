"""Cache descriptor helpers for non-attention state tensors.

BloomBee's historical inference cache descriptor describes attention KV slabs.
Qwen3.5-MoE linear-attention layers carry convolution and recurrent state
instead, so they need a descriptor that preserves the raw tensor shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch

LinearStateKind = Literal["qwen3_5_linear_conv", "qwen3_5_linear_recurrent"]


@dataclass(frozen=True)
class LinearStateTensorDescriptor:
    """Picklable descriptor for raw linear-attention state tensors.

    ``cache_allocation_tokens`` intentionally accounts by batch rows, not the
    last tensor dimension. Attention KV slabs consume memory per sequence token;
    Qwen3.5 linear states are fixed-size per request row.
    """

    kind: LinearStateKind
    size: Tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    requires_grad: bool = False

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.size

    @property
    def cache_allocation_tokens(self) -> int:
        return int(self.size[0]) if self.size else 0

    def numel(self) -> int:
        total = 1
        for dim in self.size:
            total *= int(dim)
        return int(total)


def is_linear_state_descriptor(descriptor: object) -> bool:
    return isinstance(descriptor, LinearStateTensorDescriptor)
