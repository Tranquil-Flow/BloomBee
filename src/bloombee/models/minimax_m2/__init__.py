"""BloomBee adapter for MiniMax-M2 / M2.7 REAP models."""
from transformers import AutoConfig

from bloombee.models.minimax_m2.block import WrappedMiniMaxM2Block
from bloombee.models.minimax_m2.config import DistributedMiniMaxM2Config
from bloombee.models.minimax_m2.model import DistributedMiniMaxM2ForCausalLM, DistributedMiniMaxM2Model
from bloombee.utils.auto_config import register_model_classes

try:
    AutoConfig.register("minimax_m2", DistributedMiniMaxM2Config)
except ValueError:
    pass

register_model_classes(
    config=DistributedMiniMaxM2Config,
    model=DistributedMiniMaxM2Model,
    model_for_causal_lm=DistributedMiniMaxM2ForCausalLM,
)
