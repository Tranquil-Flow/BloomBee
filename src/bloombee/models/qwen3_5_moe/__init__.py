"""BloomBee Qwen3.5-MoE text-tower registration.

This package intentionally registers the language-model text tower
(``model_type='qwen3_5_moe_text'``) only. The outer multimodal
``qwen3_5_moe`` config remains unsupported until a separate integration proves
how to unwrap/serve it safely.
"""

from transformers import AutoConfig

from bloombee.models.qwen3_5_moe.block import WrappedQwen3_5MoeTextBlock
from bloombee.models.qwen3_5_moe.config import DistributedQwen3_5MoeTextConfig
from bloombee.utils.auto_config import register_model_classes

try:
    AutoConfig.register("qwen3_5_moe_text", DistributedQwen3_5MoeTextConfig)
except ValueError:
    pass

try:
    register_model_classes(config=DistributedQwen3_5MoeTextConfig)
except AssertionError:
    # Importing both bloombee.models and this package directly should remain
    # idempotent during tests.
    pass

__all__ = ["DistributedQwen3_5MoeTextConfig", "WrappedQwen3_5MoeTextBlock"]
