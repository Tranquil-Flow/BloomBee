"""BloomBee adapter for Qwen3-MoE models (e.g. Qwen/Qwen3-30B-A3B).

Mirrors ``src/bloombee/models/qwen3/`` with two differences:

1. The base decoder layer is :class:`Qwen3MoeDecoderLayer` from
   ``transformers.models.qwen3_moe.modeling_qwen3_moe``, not the dense
   :class:`Qwen3DecoderLayer`.
2. The model registers ``model_type="qwen3_moe"`` so that ``AutoConfig``
   dispatches it through :class:`DistributedQwen3MoeConfig`.

The wrapper keeps the same fp32-rope override, the same BloomBee cache
contract (3D ``(B*H_kv, ...)``), and the same causal-mask reconstruction
as :mod:`bloombee.models.qwen3.block`.
"""

from transformers import AutoConfig

from bloombee.models.qwen3_moe.block import WrappedQwen3MoeBlock
from bloombee.models.qwen3_moe.config import DistributedQwen3MoeConfig
from bloombee.models.qwen3_moe.model import (
    DistributedQwen3MoeForCausalLM,
    DistributedQwen3MoeForSequenceClassification,
    DistributedQwen3MoeModel,
)
from bloombee.utils.auto_config import register_model_classes

# Register "qwen3_moe" model_type with HuggingFace's AutoConfig.
# transformers >= 5.x ships Qwen3MoeConfig natively; if we already beat it
# to AutoConfig.register, the second registration is a no-op.
try:
    AutoConfig.register("qwen3_moe", DistributedQwen3MoeConfig)
except ValueError:
    pass

register_model_classes(
    config=DistributedQwen3MoeConfig,
    model=DistributedQwen3MoeModel,
    model_for_causal_lm=DistributedQwen3MoeForCausalLM,
    model_for_sequence_classification=DistributedQwen3MoeForSequenceClassification,
)
