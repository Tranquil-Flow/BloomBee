import os
from typing import Optional, Union

try:
    from transformers.models.qwen3_moe import Qwen3MoeConfig as _BaseConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention as _BaseAttention
except ImportError:  # pragma: no cover - older transformers fallback
    from transformers.models.qwen3 import Qwen3Config as _BaseConfig
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention as _BaseAttention

from bloombee.client.config import ClientConfig
from bloombee.client.lm_head import LMHeadConfig
from bloombee.client.ptune import PTuneConfig
from bloombee.models.qwen3_moe.block import WrappedQwen3MoeBlock
from bloombee.utils.hivemind_compat import get_logger

logger = get_logger(__name__)


class DistributedQwen3MoeConfig(_BaseConfig, ClientConfig, PTuneConfig, LMHeadConfig):
    model_type = "qwen3_moe"

    block_class = WrappedQwen3MoeBlock
    attn_class = _BaseAttention
    block_prefix = "model.layers"

    num_key_value_groups = 1

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: Union[str, os.PathLike, None], *args, dht_prefix: Optional[str] = None, **kwargs
    ):
        loading_from_repo = model_name_or_path is not None and not os.path.isdir(model_name_or_path)
        if loading_from_repo and dht_prefix is None:
            dht_prefix = str(model_name_or_path)
            dht_prefix = dht_prefix.replace(".", "-")
            logger.info(f"Using DHT prefix: {dht_prefix}")
        result = super().from_pretrained(model_name_or_path, *args, dht_prefix=dht_prefix, **kwargs)
        config = result[0] if isinstance(result, tuple) else result
        if getattr(config, "pad_token_id", None) is None:
            config.pad_token_id = 0
        return result
