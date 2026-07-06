import contextlib
import dataclasses
import hashlib
import os
from contextvars import ContextVar
from typing import Any, ContextManager, Dict, List, Optional, Sequence, Tuple, Union

import torch
import transformers
from hivemind.utils.logging import get_logger
from torch import Tensor
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.utils import ModelOutput

from bloombee.client.inference_session import InferenceSession
from bloombee.client.remote_sequential import RemoteSequential
from bloombee.utils.misc import DUMMY, docstring_from

logger = get_logger(__name__)

# Fast-path mode: opt-in via BLOOMBEE_FAST_GENERATE=1.
#
# When eligible (plain greedy, no custom logits/stopping hooks, no beam
# search), bypasses HF generate() and runs a minimal embed → remote-forward
# → ln_f → lm_head → argmax loop. The intent is to eliminate the per-step
# Python overhead of transformers 5.x's DynamicCache update + the expanded
# prepare_inputs_for_generation + _update_model_kwargs_for_generation.
#
# Empirically (full 3×3×3 sweep on 2×A100 llama-7b/13b + falcon-7b at
# B=1/4/32, 3 trials per cell), the fast-path median ratio vs legacy HF
# generate is 0.97× — i.e. slightly slower on most cells, with occasional
# wins on specific (model, batch) combinations. The TF 5.x DynamicCache
# overhead is theoretically real but does NOT measurably dominate in
# BloomBee's distributed setup at these batch sizes; server-side compute
# and network RTT are already the bulk of step time.
#
# Left in the codebase as an opt-in experimental knob for future work
# (e.g. large-batch serving, single-GPU deployments, or if someone later
# builds a StaticCache-equivalent for distributed KV).
_FAST_GENERATE_ENABLED = os.environ.get("BLOOMBEE_FAST_GENERATE", "0") == "1"


class RemotePastKeyValues(Cache):
    """only keeps the number of seen tokens. pretends to be a legit cache"""

    def __init__(self) -> None:
        # TF 5.x's Cache.__init__ requires exactly one of `layers` or
        # `layer_class_to_replicate`. BloomBee doesn't actually store layers
        # here (the real cache lives on the remote server), so hand it an
        # empty list. TF 4.x's Cache.__init__ takes no args — in that case we
        # skip super() entirely to stay compatible.
        try:
            super().__init__(layers=[])
        except TypeError:
            super().__init__()
        self._seen_tokens: Optional[torch.Tensor] = None
        self.hypo_ids: Optional[torch.LongTensor] = None
        self.kv_cache_position_ids: Optional[torch.LongTensor] = None
        self.is_spec_decoding: Optional[torch.LongTensor] = None
        self.prefill_length: Optional[torch.LongTensor] = None

    def __getitem__(self, _index: int) -> List[torch.Tensor]:
        return [DUMMY]  # For compatibility with BloomForCausalLM.prepare_inputs_for_generation()

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if self._seen_tokens is None:
            return 0
        if self._seen_tokens.dim() == 0:
            return self._seen_tokens.item()
        return self._seen_tokens[0].item()
    
    def get_seq_length_batch(self) -> Optional[torch.Tensor]:
        return self._seen_tokens

    def get_max_length(self) -> Optional[int]:
        return None

    def update_seen(self, new_seen: Union[int, torch.Tensor]) -> None:
        if isinstance(new_seen, int):
            self._seen_tokens = torch.tensor([new_seen])
        elif isinstance(new_seen, torch.Tensor):
            if new_seen.dim() == 0:
                new_seen = new_seen.unsqueeze(0)
            self._seen_tokens = new_seen
        else:
            raise TypeError(f"new_seen must be int or torch.Tensor, got {type(new_seen)}")


    def reorder_cache(self, beam_idx):
        raise NotImplementedError("Beam search reordering is not implemented yet")
    
    def set_kv_cache(self, position_ids: Optional[torch.LongTensor]):
        self.kv_cache_position_ids = position_ids
        
    def set_is_spec_decoding(self, is_spec_decoding: Optional[torch.LongTensor]):
        self.is_spec_decoding = is_spec_decoding
        
    def set_prefill_length(self, prefill_length: Optional[torch.LongTensor]):
        self.prefill_length = prefill_length


_skipped_tokens = ContextVar("skipped_tokens", default=0)


def _logits_sha256_per_row(logits: torch.Tensor) -> list[str]:
    """Return stable per-row fingerprints for logits used by live parity evidence."""
    if logits.ndim == 3:
        logits = logits[:, -1, :]
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, vocab] or [batch, 1, vocab]")
    rows = logits.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return [hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in rows]


class _SkipTokensMixin:
    # This override is used in RemoteGenerationMixin by has to be defined in a class not named as "GenerationMixin"
    # due to how transformers.PreTrainedModel.can_generate() works
    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs) -> dict:
        input_ids = input_ids[:, _skipped_tokens.get() :]
        _skipped_tokens.set(0)
        return super().prepare_inputs_for_generation(input_ids, **kwargs)


class RemoteGenerationMixin(_SkipTokensMixin):
    """
    This class is an upgrade to `transformers.GenerationMixin` that:

    - Designed to be compatible with most `transformers.GenerationMixin` strategies and options
    - Supports generation inside a remote InferenceSession, so that remote servers store your attention caches and
      you don't have to rerun the prefix through all the servers to generate each new token
    - Supports multiple `.generate()` calls inside one InferenceSession, so you can easily run interactive generation
      by showing tokens on the fly (multiple calls like `.generate(None, max_new_tokens=1, ...)`) or
      accept prompts from a user in a chat bot (multiple calls like `.generate(new_prompts, ...)`).
    - If there is no active session, `.generate()` will create a new InferenceSession with proper `max_length`.
      Otherwise, `.generate()` will use the active session. You can use the `session=...` argument to override that.
    """

    @docstring_from(RemoteSequential.active_session)
    @property
    def active_session(self) -> Optional[InferenceSession]:
        return self.transformer.h.active_session

    @docstring_from(RemoteSequential.use_session)
    def use_session(self, session: Optional[InferenceSession]) -> ContextManager[InferenceSession]:
        return self.transformer.h.use_session(session)

    @docstring_from(RemoteSequential.inference_session)
    def inference_session(self, **kwargs) -> ContextManager[InferenceSession]:
        return self.transformer.h.inference_session(**kwargs)

    @docstring_from(transformers.GenerationMixin.generate.__doc__)
    def generate(
        self, inputs: Optional[torch.Tensor] = None, *args, session: Optional[InferenceSession] = None, **kwargs
    ):
        self._fix_generate_kwargs(kwargs)
        if inputs is None:
            inputs = kwargs.pop("input_ids", None)

        # Live-continuous batching seam: deliberately opt-in and conservative.
        # The first production wiring records LiveDecodeRow ticks on the
        # InferenceSession for plain greedy single-request decode; unsupported
        # modes fall through unchanged.
        if self._live_continuous_generate_eligible(inputs, args, kwargs, session):
            return self._live_continuous_generate_impl(inputs, *args, session=session, **kwargs)

        # Fast-path: bypass HF GenerationMixin for plain greedy decoding.
        # Saves ~8-19 ms per decode step at B=32 on transformers 5.x by
        # skipping DynamicCache.update, prepare_inputs_for_generation,
        # _update_model_kwargs_for_generation, logits_processor/stopping_criteria
        # dispatch, and several layers of *args/**kwargs plumbing. Falls back
        # to super().generate() when the request uses any feature the fast
        # path does not support (sampling, beam search, custom logits processors,
        # attention_mask, spec-decoding, etc.).
        if self._fast_generate_eligible(inputs, args, kwargs, session):
            return self._fast_generate_greedy(inputs, session=session, **kwargs)

        if session is not None:
            # If a session specified explicitly, use it
            context_manager = self.use_session(session)
        elif self.active_session is not None:
            # If there's an active session, don't do anything
            context_manager = contextlib.nullcontext(self.active_session)
        else:
            # If there's no active session, create a new one

            max_length = kwargs.get("max_length")
            max_new_tokens = kwargs.get("max_new_tokens")
            assert (max_length is None) != (
                max_new_tokens is None
            ), "You should set `max_length` or `max_new_tokens` (but not both) to reserve server-side attention caches"

            session_max_length = self.transformer.config.pre_seq_len
            if max_length is not None:
                session_max_length += max_length
            else:
                session_max_length += (inputs.shape[1] if inputs is not None else 0) + max_new_tokens
            context_manager = self.inference_session(max_length=session_max_length)

        with context_manager as session:
            # Prepend the tokens from the previous .generate() call
            n_prev_tokens = session.output_ids.shape[1] if session.output_ids is not None else 0
            if n_prev_tokens > 0:
                if kwargs.get("num_beams", 1) > 1:
                    logger.warning(
                        "Beam search will not work properly in the resumed petals.InferenceSession "
                        "since intermediate beam entries are lost"
                    )

                if inputs is not None:
                    inputs = torch.cat([session.output_ids, inputs], dim=1)
                else:
                    inputs = session.output_ids

                # Don't actually run all previous tokens through the transformer,
                # but keep them for transformers.GenerationMixin (e.g., to compute repetition_penalty)
                _skipped_tokens.set(max(0, n_prev_tokens - 1))

            if self._supports_cache_class and "past_key_values" not in kwargs:
                past_key_values = RemotePastKeyValues()
                past_key_values.update_seen(session.position)
                kwargs["past_key_values"] = past_key_values

            result = super().generate(inputs, *args, **kwargs)

            sequences = result.sequences if isinstance(result, ModelOutput) else result
            # Save tokens from this .generate() call
            session.output_ids = sequences
            # Crop the last tokens from the previous call
            sequences = sequences[:, n_prev_tokens:].clone()
            if isinstance(result, ModelOutput):
                result.sequences = sequences
            else:
                result = sequences

        return result

    # ------------------------------------------------------------------
    # Fast-path greedy generate (bypasses HF GenerationMixin)
    # ------------------------------------------------------------------
    # The "why" is documented on the BLOOMBEE_FAST_GENERATE flag at the top
    # of this file. The "what" is: an equivalent greedy decode that calls
    # the same client modules (word_embeddings, layers=RemoteSequential,
    # ln_f, lm_head) but skips transformers' generate machinery.

    _LIVE_CONTINUOUS_KNOWN_KWARGS = frozenset(
        {
            "max_length", "max_new_tokens", "do_sample", "pad_token_id",
            "eos_token_id", "bos_token_id", "use_cache", "num_beams",
            "num_return_sequences", "return_dict_in_generate", "attention_mask",
            "logits_processor", "stopping_criteria", "generation_config", "live_arrival_ticks",
        }
    )

    def _live_continuous_generate_eligible(
        self,
        inputs: Optional[torch.Tensor],
        args: tuple,
        kwargs: dict,
        session: Optional[InferenceSession],
    ) -> bool:
        try:
            from bloombee.client.live_continuous_batching import is_live_continuous_batching_enabled
        except Exception:
            return False
        if not is_live_continuous_batching_enabled():
            return False
        if not callable(getattr(self, "_live_continuous_generate_impl", None)):
            return False
        if inputs is None or not isinstance(inputs, torch.Tensor) or inputs.ndim != 2:
            return False
        if inputs.shape[0] < 1:
            return False
        if inputs.shape[0] > 1 and type(self)._live_continuous_generate_impl is not RemoteGenerationMixin._live_continuous_generate_impl:
            return False
        max_batch_raw = os.environ.get("BLOOMBEE_LIVE_CONTINUOUS_MAX_BATCH_SIZE")
        if max_batch_raw:
            try:
                if inputs.shape[0] > int(max_batch_raw):
                    return False
            except ValueError:
                return False
        if args:
            return False
        if self.active_session is not None and session is not self.active_session:
            return False
        if session is not None and (
            not callable(getattr(session, "stage_live_continuous_tick_rows", None))
            or not callable(getattr(session, "record_live_continuous_tick_rows", None))
        ):
            return False
        if kwargs.get("do_sample", False):
            return False
        if kwargs.get("num_beams", 1) != 1:
            return False
        if kwargs.get("num_return_sequences", 1) != 1:
            return False
        if kwargs.get("attention_mask") is not None:
            return False
        if "logits_processor" in kwargs and kwargs["logits_processor"]:
            return False
        if "stopping_criteria" in kwargs and kwargs["stopping_criteria"]:
            return False
        if kwargs.get("return_dict_in_generate"):
            return False
        if kwargs.get("generation_config") is not None:
            return False
        unknown = set(kwargs) - self._LIVE_CONTINUOUS_KNOWN_KWARGS
        if unknown:
            return False
        if kwargs.get("max_new_tokens") is None and kwargs.get("max_length") is None:
            return False
        if getattr(self.transformer.config, "tuning_mode", None):
            return False
        return True

    @torch.inference_mode()
    def _live_continuous_generate_impl(
        self,
        input_ids: torch.Tensor,
        *args,
        session: Optional[InferenceSession] = None,
        max_new_tokens: Optional[int] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        bos_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        live_arrival_ticks: Optional[Sequence[int]] = None,
        **_ignored,
    ) -> torch.Tensor:
        """Minimal opt-in live-continuous request loop for conservative greedy decode.

        This wires ``LiveDecodeRow`` telemetry into the active
        ``InferenceSession`` while using the same client request path as the
        fast greedy loop (embedding -> RemoteSequential/session.step -> ln_f ->
        lm_head). It supports same-arrival greedy batches behind the opt-in flag;
        late-arrival queueing still lives in the separate proof harness, and the
        session report remains claim-bounded.
        """
        if args:
            raise RuntimeError("live continuous batching does not accept positional generation args")
        if self.active_session is not None and session is not self.active_session:
            raise RuntimeError("live continuous batching requires a fresh or explicitly active inference session")
        if input_ids.ndim != 2 or input_ids.shape[0] < 1:
            raise RuntimeError("live continuous batching expects rank-2 input_ids with at least one row")

        from bloombee.client.live_continuous_batching import LiveDecodeRow

        batch_size, prompt_len = input_ids.shape
        if max_new_tokens is None:
            assert max_length is not None
            max_new_tokens = max(0, int(max_length) - prompt_len)
        else:
            max_new_tokens = int(max_new_tokens)
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")

        eos_set: Tuple[int, ...]
        if isinstance(eos_token_id, int):
            eos_set = (eos_token_id,)
        elif isinstance(eos_token_id, (list, tuple)):
            eos_set = tuple(int(x) for x in eos_token_id)
        else:
            eos_set = ()

        pre_seq_len = getattr(self.transformer.config, "pre_seq_len", 0) or 0
        if session is not None:
            context_manager = self.use_session(session)
        else:
            context_manager = self.inference_session(max_length=pre_seq_len + prompt_len + max_new_tokens)

        normalized_arrival_ticks: list[int] | None = None
        if live_arrival_ticks is not None:
            normalized_arrival_ticks = [int(tick) for tick in live_arrival_ticks]
            if len(normalized_arrival_ticks) != batch_size:
                raise ValueError("live_arrival_ticks length must match input batch size")
            if any(tick < 0 for tick in normalized_arrival_ticks):
                raise ValueError("live_arrival_ticks must be non-negative")
            if any(tick > 0 for tick in normalized_arrival_ticks):
                if prompt_len != 1:
                    raise RuntimeError(
                        "late-arrival live continuous batching currently requires one-token prompt rows; "
                        "multi-token ragged prefill remains fail-closed"
                    )
                return self._live_continuous_generate_late_arrival_impl(
                    input_ids,
                    context_manager=context_manager,
                    max_new_tokens=max_new_tokens,
                    eos_set=eos_set,
                    arrival_ticks=normalized_arrival_ticks,
                )

        embed = self.transformer.word_embeddings
        layers = self.transformer.h
        ln_f = self.transformer.ln_f
        lm_head = self.lm_head

        output = input_ids
        step_ids = input_ids
        done = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        with context_manager as live_session:
            recorder = getattr(live_session, "record_live_continuous_tick_rows", None)
            stager = getattr(live_session, "stage_live_continuous_tick_rows", None)
            if not callable(recorder) or not callable(stager):
                raise RuntimeError("InferenceSession cannot stage and record live continuous batching rows")

            request_ids = [f"generate-{batch_idx}" for batch_idx in range(batch_size)]
            try:
                from bloombee.client.kv_prefix_reuse import is_kv_prefix_reuse_enabled
            except Exception:
                is_kv_prefix_reuse_enabled = lambda: False  # type: ignore[assignment]
            if batch_size >= 2 and is_kv_prefix_reuse_enabled():
                kv_recorder = getattr(live_session, "record_kv_prefix_reuse_prefill", None)
                if callable(kv_recorder):
                    try:
                        kv_recorder(input_ids, request_ids=request_ids)
                    except ValueError:
                        # Not every live-continuous batch has a reusable prefix;
                        # keep generation working and simply omit KV-prefix metadata.
                        pass

            for tick in range(max_new_tokens):
                row_input_ids = [int(token.detach().cpu().item()) for token in step_ids[:, -1]]
                rows = [
                    LiveDecodeRow(
                        request_id=request_ids[batch_idx],
                        tick=tick,
                        position=tick,
                        input_token_id=row_input_id,
                    )
                    for batch_idx, row_input_id in enumerate(row_input_ids)
                ]
                stager(rows)
                hidden = embed(step_ids)
                hidden = layers(hidden)
                hidden = ln_f(hidden)
                logits = lm_head(hidden[:, -1:, :])
                next_id = logits.argmax(dim=-1)

                if eos_set:
                    for eos_id in eos_set:
                        done = done | next_id.squeeze(-1).eq(eos_id)

                recorder(
                    rows,
                    output_token_ids=[int(token.detach().cpu().item()) for token in next_id[:, 0]],
                    output_logits_sha256=_logits_sha256_per_row(logits),
                )

                output = torch.cat([output, next_id], dim=1)
                step_ids = next_id

                if eos_set and bool(done.all().item()):
                    break

            live_session.output_ids = output

        return output

    @torch.inference_mode()
    def _live_continuous_generate_late_arrival_impl(
        self,
        input_ids: torch.Tensor,
        *,
        context_manager: ContextManager,
        max_new_tokens: int,
        eos_set: Tuple[int, ...],
        arrival_ticks: Sequence[int],
    ) -> torch.Tensor:
        """Token-level live-continuous scheduler for staggered one-token rows.

        This path is intentionally narrow. It proves a real late-arrival merge
        seam for already-tokenized one-token prompts without pretending to solve
        ragged multi-token prefill.
        """
        from collections import deque
        from bloombee.client.live_continuous_batching import LiveDecodeRow

        embed = self.transformer.word_embeddings
        layers = self.transformer.h
        ln_f = self.transformer.ln_f
        lm_head = self.lm_head

        batch_size = int(input_ids.shape[0])
        generated_by_row: list[list[int]] = [[] for _ in range(batch_size)]
        done = [False for _ in range(batch_size)]
        pending = deque(sorted(range(batch_size), key=lambda idx: (int(arrival_ticks[idx]), idx)))
        active: deque[int] = deque()
        active_set: set[int] = set()
        tick = min(int(tick_value) for tick_value in arrival_ticks) if arrival_ticks else 0

        with context_manager as live_session:
            setattr(live_session, "_live_continuous_full_batch_size", batch_size)
            recorder = getattr(live_session, "record_live_continuous_tick_rows", None)
            stager = getattr(live_session, "stage_live_continuous_tick_rows", None)
            if not callable(recorder) or not callable(stager):
                raise RuntimeError("InferenceSession cannot stage and record live continuous batching rows")

            request_ids = [f"generate-{batch_idx}" for batch_idx in range(batch_size)]
            while any(len(tokens) < max_new_tokens for tokens in generated_by_row):
                while pending and int(arrival_ticks[pending[0]]) <= tick:
                    row_idx = pending.popleft()
                    if row_idx not in active_set and not done[row_idx]:
                        active.append(row_idx)
                        active_set.add(row_idx)
                if not active:
                    if pending:
                        tick = max(tick + 1, int(arrival_ticks[pending[0]]))
                        continue
                    break

                row_indices: list[int] = []
                max_batch_raw = os.environ.get("BLOOMBEE_LIVE_CONTINUOUS_MAX_BATCH_SIZE")
                max_batch = batch_size
                if max_batch_raw:
                    try:
                        max_batch = max(1, min(batch_size, int(max_batch_raw)))
                    except ValueError:
                        max_batch = batch_size
                for _ in range(max_batch):
                    if not active:
                        break
                    row_idx = active.popleft()
                    active_set.remove(row_idx)
                    if not done[row_idx]:
                        row_indices.append(row_idx)
                if not row_indices:
                    tick += 1
                    continue

                input_tokens = []
                rows = []
                for row_idx in row_indices:
                    position = len(generated_by_row[row_idx])
                    token_id = int(input_ids[row_idx, -1].detach().cpu().item()) if position == 0 else generated_by_row[row_idx][-1]
                    input_tokens.append(token_id)
                    rows.append(
                        LiveDecodeRow(
                            request_id=request_ids[row_idx],
                            tick=tick,
                            position=position,
                            input_token_id=token_id,
                        )
                    )

                stager(rows)
                step_ids = torch.tensor(input_tokens, dtype=input_ids.dtype, device=input_ids.device).unsqueeze(1)
                hidden = embed(step_ids)
                hidden = layers(hidden)
                if hidden.shape[0] < len(row_indices):
                    raise RuntimeError(
                        "live continuous batching server returned fewer rows than the active tick batch"
                    )
                if hidden.shape[0] > len(row_indices):
                    hidden = hidden[: len(row_indices)]
                hidden = ln_f(hidden)
                logits = lm_head(hidden[:, -1:, :])
                next_id = logits.argmax(dim=-1)
                output_tokens = [int(token.detach().cpu().item()) for token in next_id[:, 0]]
                recorder(
                    rows,
                    output_token_ids=output_tokens,
                    output_logits_sha256=_logits_sha256_per_row(logits),
                )

                for row_idx, token_id in zip(row_indices, output_tokens):
                    generated_by_row[row_idx].append(int(token_id))
                    if eos_set and token_id in eos_set:
                        done[row_idx] = True
                    if not done[row_idx] and len(generated_by_row[row_idx]) < max_new_tokens:
                        active.append(row_idx)
                        active_set.add(row_idx)

                tick += 1

            output_rows = []
            for row_idx in range(batch_size):
                generated = generated_by_row[row_idx]
                if len(generated) < max_new_tokens:
                    generated = generated + [int(input_ids[row_idx, -1].detach().cpu().item())] * (max_new_tokens - len(generated))
                output_rows.append(
                    torch.cat(
                        [input_ids[row_idx], torch.tensor(generated[:max_new_tokens], dtype=input_ids.dtype, device=input_ids.device)]
                    )
                )
            output = torch.stack(output_rows, dim=0)
            live_session.output_ids = output
        return output

    # Kwargs that the fast path consumes or can safely ignore. Anything
    # else (e.g. attention_mask, logits_processor, stopping_criteria,
    # generation_config with non-default fields, num_beams>1, do_sample)
    # forces the legacy path.
    _FAST_GENERATE_KNOWN_KWARGS = frozenset(
        {
            "max_length", "max_new_tokens", "do_sample", "pad_token_id",
            "eos_token_id", "bos_token_id", "use_cache", "output_hidden_states",
            "output_attentions", "return_dict_in_generate", "past_key_values",
        }
    )

    def _fast_generate_eligible(
        self,
        inputs: Optional[torch.Tensor],
        args: tuple,
        kwargs: dict,
        session: Optional[InferenceSession],
    ) -> bool:
        if not _FAST_GENERATE_ENABLED:
            return False
        if inputs is None or not isinstance(inputs, torch.Tensor) or inputs.ndim != 2:
            return False
        if args:
            # Positional args besides `inputs` are not standard; fall back.
            return False
        if kwargs.get("do_sample", False):
            return False
        if kwargs.get("num_beams", 1) != 1:
            return False
        if kwargs.get("num_return_sequences", 1) != 1:
            return False
        if kwargs.get("attention_mask") is not None:
            # HF's default handles left-padding via attention_mask. Our fast
            # path doesn't; ship it through the legacy generator for now.
            return False
        if "logits_processor" in kwargs and kwargs["logits_processor"]:
            return False
        if "stopping_criteria" in kwargs and kwargs["stopping_criteria"]:
            return False
        if kwargs.get("output_hidden_states") or kwargs.get("output_attentions"):
            return False
        if kwargs.get("return_dict_in_generate"):
            return False
        if kwargs.get("generation_config") is not None:
            # Custom generation_config may carry flags we don't honor.
            return False
        # Unknown kwargs → legacy path.
        unknown = set(kwargs) - self._FAST_GENERATE_KNOWN_KWARGS - {"generation_config"}
        if unknown:
            return False
        # max_length or max_new_tokens must be set.
        if kwargs.get("max_new_tokens") is None and kwargs.get("max_length") is None:
            return False
        # pTune prefix tokens require the legacy forward; skip fast path.
        if getattr(self.transformer.config, "tuning_mode", None):
            return False
        return True

    @torch.inference_mode()
    def _fast_generate_greedy(
        self,
        input_ids: torch.Tensor,
        *,
        session: Optional[InferenceSession] = None,
        max_new_tokens: Optional[int] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        bos_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[Cache] = None,
        **_ignored,
    ) -> torch.Tensor:
        """Greedy decode loop that avoids HF GenerationMixin overhead.

        Returns a ``torch.Tensor`` of shape ``(batch, prompt_len + n_new)``,
        matching the default ``generate()`` output when
        ``return_dict_in_generate=False``.
        """
        assert input_ids.ndim == 2, "fast generate expects (batch, seq_len) input_ids"
        batch_size, prompt_len = input_ids.shape

        if max_new_tokens is None:
            assert max_length is not None
            max_new_tokens = max(0, int(max_length) - prompt_len)

        # Normalize eos to a set of ints for fast membership test.
        eos_set: Tuple[int, ...]
        if isinstance(eos_token_id, int):
            eos_set = (eos_token_id,)
        elif isinstance(eos_token_id, (list, tuple)):
            eos_set = tuple(int(x) for x in eos_token_id)
        else:
            eos_set = ()

        # Session bring-up mirrors the legacy path: if one isn't supplied,
        # open a new one sized for `prompt_len + max_new_tokens` plus the
        # p-tune prefix (pre_seq_len=0 on all checkpoints we currently serve).
        context_manager: ContextManager[InferenceSession]
        if session is not None:
            context_manager = self.use_session(session)
        elif self.active_session is not None:
            context_manager = contextlib.nullcontext(self.active_session)
        else:
            pre_seq_len = getattr(self.transformer.config, "pre_seq_len", 0) or 0
            context_manager = self.inference_session(
                max_length=pre_seq_len + prompt_len + max_new_tokens
            )

        # Shorthand bindings.
        embed = self.transformer.word_embeddings  # nn.Embedding on client
        layers = self.transformer.h               # RemoteSequential
        ln_f = self.transformer.ln_f              # final norm on client
        lm_head = self.lm_head                    # projection to vocab on client

        output = input_ids
        done = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        with context_manager as _sess:
            # Resume-token handling (matches the legacy path's n_prev_tokens logic).
            if _sess.output_ids is not None and _sess.output_ids.shape[1] > 0:
                # Session already advanced past some tokens; only the newest
                # token needs to be fed on this step.
                prev = _sess.output_ids
                output = torch.cat([prev, input_ids], dim=1) if input_ids.shape[1] > 0 else prev
                step_ids = input_ids[:, -1:] if input_ids.shape[1] > 0 else prev[:, -1:]
            else:
                step_ids = input_ids  # Prefill with the full prompt on step 0.

            for step in range(max_new_tokens):
                hidden = embed(step_ids)               # (B, step_tokens, H)
                hidden = layers(hidden)                # RemoteSequential → session.step
                hidden = ln_f(hidden)                  # (B, step_tokens, H)
                logits = lm_head(hidden[:, -1:, :])    # (B, 1, V) — only last position
                # Greedy argmax. logits.dtype may be fp16/bf16; argmax is
                # dtype-agnostic so no cast needed.
                next_id = logits.argmax(dim=-1)        # (B, 1)

                if eos_set:
                    # Once a sequence has emitted EOS, keep it frozen on
                    # pad_token_id (or EOS when pad is missing) — matches
                    # HF's behavior under `pad_token_id`.
                    for e in eos_set:
                        done = done | next_id.squeeze(-1).eq(e)
                    if pad_token_id is not None:
                        next_id = torch.where(
                            done.unsqueeze(-1), torch.full_like(next_id, pad_token_id), next_id
                        )

                output = torch.cat([output, next_id], dim=1)
                step_ids = next_id

                if eos_set and bool(done.all().item()):
                    break

            # Keep the session's output_ids consistent with the legacy path.
            _sess.output_ids = output

        return output

    @staticmethod
    def _fix_generate_kwargs(kwargs: dict):
        # Suppress inappropriate "Both max_new_tokens and max_length" HF warning
        if "max_length" in kwargs and kwargs["max_length"] is None:
            del kwargs["max_length"]

        # Support do_sample = {0, 1} for backward compatibility with Petals < 2.1.0
        do_sample = kwargs.get("do_sample")
        if isinstance(do_sample, int):
            kwargs["do_sample"] = bool(do_sample)

    @staticmethod
    def _reorder_cache(past_key_values: RemotePastKeyValues, beam_idx: torch.LongTensor) -> RemotePastKeyValues:
        return dataclasses.replace(past_key_values, hypo_ids=beam_idx)
