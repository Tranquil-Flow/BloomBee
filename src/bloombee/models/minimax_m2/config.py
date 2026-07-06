import os
from typing import Optional, Union

from transformers.models.minimax_m2 import MiniMaxM2Config as _BaseConfig
from transformers.models.minimax_m2.modeling_minimax_m2 import MiniMaxM2Attention as _BaseAttention

from bloombee.client.config import ClientConfig
from bloombee.client.lm_head import LMHeadConfig
from bloombee.client.ptune import PTuneConfig
from bloombee.models.minimax_m2.block import WrappedMiniMaxM2Block
from bloombee.utils.hivemind_compat import get_logger

logger = get_logger(__name__)


class DistributedMiniMaxM2Config(_BaseConfig, ClientConfig, PTuneConfig, LMHeadConfig):
    model_type = "minimax_m2"

    block_class = WrappedMiniMaxM2Block
    attn_class = _BaseAttention
    block_prefix = "model.layers"

    num_key_value_groups = 1

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: Union[str, os.PathLike, None], *args, dht_prefix: Optional[str] = None, **kwargs
    ):
        loading_from_repo = model_name_or_path is not None and not os.path.isdir(model_name_or_path)
        if loading_from_repo and dht_prefix is None:
            dht_prefix = str(model_name_or_path).replace(".", "-")
            logger.info(f"Using DHT prefix: {dht_prefix}")
        result = super().from_pretrained(model_name_or_path, *args, dht_prefix=dht_prefix, **kwargs)
        config = result[0] if isinstance(result, tuple) else result
        if getattr(config, "pad_token_id", None) is None:
            config.pad_token_id = 0
        return result
