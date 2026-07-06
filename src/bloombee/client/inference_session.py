from __future__ import annotations

import asyncio
import itertools
import time
import uuid
from typing import Any, AsyncIterator, List, Optional, Tuple

import torch
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from hivemind.p2p import P2P
from hivemind.proto import runtime_pb2
from hivemind.utils.tensor_descr import BatchTensorDescriptor

from bloombee.client.config import ClientConfig
from bloombee.client.routing import RemoteSequenceManager, maybe_log_traceback
from bloombee.data_structures import CHAIN_DELIMITER, ModuleUID, RemoteSpanInfo, RPCInfo
from bloombee.server.handler import TransformerConnectionHandler
from bloombee.utils.hivemind_compat import MSGPackSerializer, anext, get_logger
from bloombee.utils.debug_config import is_log_channel_enabled
from bloombee.utils.lossless_transport import (
    deserialize_torch_tensor,
    serialize_torch_tensor,
    transport_profile_scope,
    log_transport_profile_event,
)
from bloombee.utils.misc import DUMMY, DUMMY_INT64, is_dummy
from bloombee.utils.packaging import normalize_arg
from bloombee.utils.real_activation_dumper import capture_wire_activation
from bloombee.utils.microbatch_config import (
    is_microbatch_enabled,
    get_micro_batch_size,
    get_current_path,
    log_config as mbpipe_log_config,
    log_path_entry as mbpipe_log_path_entry,
    MBPIPE_LOG_PREFIX,
)

logger = get_logger(__name__)


_FLOATING_WIRE_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}


def _dtype_name(dtype: Optional[torch.dtype]) -> str:
    return "" if dtype is None else str(dtype).replace("torch.", "")


def _is_floating_wire_dtype(dtype: Optional[torch.dtype]) -> bool:
    return dtype in _FLOATING_WIRE_DTYPES


def _server_hidden_states_wire_dtype(server_side_inference_schema) -> Optional[torch.dtype]:
    try:
        dtype = server_side_inference_schema[0].dtype
    except Exception:
        return None
    return dtype if _is_floating_wire_dtype(dtype) else None


def _server_session_tokens_to_advance(sent_inputs: torch.Tensor, current_step_tokens: int, is_spec_dec: bool) -> int:
    """Return how many tokens a per-server session should add to its cache position.

    Existing decode sessions send only the new token. Replacement sessions after
    an RPC failure are unstepped and send their full hidden-state history to
    rebuild remote KV cache.  In that recovery case, advancing by only the
    current token leaves the per-server session at position 1 even though the
    server rebuilt the whole prefix, causing later decode requests to use a
    stale prefix.
    """
    if is_spec_dec:
        return int(current_step_tokens)
    if torch.is_tensor(sent_inputs) and sent_inputs.ndim >= 2:
        return int(sent_inputs.shape[1])
    return int(current_step_tokens)


def _trim_recovered_history_for_existing_downstream(
    inputs: torch.Tensor,
    current_step_tokens: int,
    downstream_position: int,
    is_spec_dec: bool,
) -> torch.Tensor:
    """Trim full-history recovery output before an already-warm downstream stage.

    If an upstream stage is recreated after failure, it resends full hidden-state
    history to rebuild its own KV cache and may return that full history. Later
    stages that did *not* fail already have the prefix cached; feeding the full
    history to them again makes their server-side guard see e.g. ``prefix=6`` +
    ``current=7`` for a session with ``max_length=12``. Replacement downstream
    sessions have ``downstream_position == 0`` and must keep the full history.
    """
    if is_spec_dec or current_step_tokens <= 0 or downstream_position <= 0:
        return inputs
    if not torch.is_tensor(inputs) or inputs.ndim < 2:
        return inputs
    if int(inputs.shape[1]) <= int(current_step_tokens):
        return inputs
    return inputs[:, -int(current_step_tokens):, :].contiguous()


def _recovery_error_code(error: BaseException) -> str:
    """Classify noisy recovery exceptions into stable log-scrapable reasons."""
    text = f"{type(error).__name__}: {error!s} {error!r}".lower()
    if "placeholder storage has not been allocated on mps device" in text:
        return "mps_placeholder_storage"
    if "maximum length" in text or "max_length" in text:
        return "cache_length_mismatch"
    if "failed to call handler" in text or "p2phandlererror" in text:
        return "rpc_handler_error"
    return type(error).__name__


def _format_recovery_retry_event(
    span,
    attempt_no: int,
    max_retries: int,
    delay_s: float,
    error: BaseException,
) -> str:
    return (
        "[RECOVERY_EVENT] type=rpc_inference_retry action=retry "
        f"reason={_recovery_error_code(error)} "
        f"attempt={attempt_no + 1}/{max_retries} "
        f"delay_s={delay_s:.0f} span={span} error={error!r}"
    )


def _format_final_history_trim_event(
    seq_len: int,
    current_step_tokens: int,
    client_position: int,
) -> str:
    return (
        "[RECOVERY_EVENT] type=final_history_trim "
        "reason=session_rebuild_full_history action=trim_to_current_window "
        f"seq_len={seq_len} current_step_tokens={current_step_tokens} "
        f"client_position={client_position}"
    )


def _prepare_rpc_inference_tensor_for_wire(
    tensor: torch.Tensor,
    tensor_name: str,
    compression: runtime_pb2.CompressionType,
    server_hidden_states_dtype: Optional[torch.dtype],
) -> Tuple[torch.Tensor, BatchTensorDescriptor, dict]:
    original_dtype = tensor.dtype if torch.is_tensor(tensor) else None
    schema_dtype = server_hidden_states_dtype if tensor_name == "hidden_states" else None
    target_dtype = original_dtype
    dtype_guard_applied = 0

    if (
        tensor_name == "hidden_states"
        and torch.is_tensor(tensor)
        and torch.is_floating_point(tensor)
        and _is_floating_wire_dtype(schema_dtype)
    ):
        target_dtype = schema_dtype
        dtype_guard_applied = int(original_dtype != target_dtype)

    wire_tensor = tensor
    if torch.is_tensor(wire_tensor):
        if target_dtype is not None and wire_tensor.dtype != target_dtype:
            wire_tensor = wire_tensor.to(target_dtype)
        if not wire_tensor.is_contiguous():
            wire_tensor = wire_tensor.contiguous()

    proto = BatchTensorDescriptor.from_tensor(wire_tensor, compression)
    debug_fields = {
        "compute_dtype": _dtype_name(original_dtype),
        "schema_dtype": _dtype_name(schema_dtype),
        "wire_dtype": _dtype_name(wire_tensor.dtype if torch.is_tensor(wire_tensor) else None),
        "dtype_guard_applied": dtype_guard_applied,
        "upcast_suspect": int(
            tensor_name == "hidden_states"
            and original_dtype == torch.float32
            and schema_dtype in (torch.float16, torch.bfloat16)
        ),
    }
    if dtype_guard_applied:
        debug_fields["wire_cast"] = f"{_dtype_name(original_dtype)}_to_{_dtype_name(target_dtype)}_server_schema"
    return wire_tensor, proto, debug_fields


class _ServerInferenceSession:
    """
    An interface to a single multi-step *inference* session for a a set of blocks on a specific server.

    :note: This class is *not* fault-tolerant out of the box.
    """

    def __init__(
        self,
        config: ClientConfig,
        span: RemoteSpanInfo,
        uid: ModuleUID,
        rpc_info: RPCInfo,
        inputs_queue: asyncio.Queue,
        outputs_aiter: AsyncIterator,
        *,
        max_length: int,
        **metadata,
    ):
        self.config = config
        self.span, self.uid, self.rpc_info = span, uid, rpc_info
        self.num_blocks = uid.count(CHAIN_DELIMITER) + 1
        self._inputs_queue: asyncio.Queue[runtime_pb2.ExpertRequest] = inputs_queue
        self._outputs_stream: AsyncIterator[runtime_pb2.ExpertResponse] = outputs_aiter
        self.session_id = str(uuid.uuid4())
        self.session_metadata = dict(max_length=max_length, **metadata)
        self.stepped = False
        self.closed = False

        self._position = 0
        self.history = None  # Used in case of server failures to regenerate attention caches on new servers
        self.next_session = None
        self._pending_live_continuous_tick_batch = None
        self._server_response_metadata_events = []

    @classmethod
    async def create(
        cls,
        config: ClientConfig,
        p2p: P2P,
        span: RemoteSpanInfo,
        uid: ModuleUID,
        rpc_info: RPCInfo,
        **metadata,
    ) -> _ServerInferenceSession:
        """Create a new session for a given remote module. This code is meant to be run inside RemoteExpertWorker"""
        stub = TransformerConnectionHandler.get_stub(p2p, span.peer_id)
        inputs_queue = asyncio.Queue()
        outputs_stream = await asyncio.wait_for(
            stub.rpc_inference(cls._read_inputs_from_queue(inputs_queue)),
            config.connect_timeout,
        )
        return cls(config, span, uid, rpc_info, inputs_queue, outputs_stream, **metadata)

    @staticmethod
    async def _read_inputs_from_queue(queue: asyncio.Queue, input_timeout: Optional[float] = None) -> AsyncIterator:
        while True:
            next_input_message = await asyncio.wait_for(queue.get(), input_timeout)
            yield next_input_message
            if not next_input_message.uid and not next_input_message.tensors:
                break  # this message means "done sending"

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, start_from_position: int):
        # assert start_from_position <= self._position
        self._position = start_from_position
        if self.history is not None and self.history.shape[1] >= start_from_position:
            self.history = self.history[:, :start_from_position, :] if start_from_position > 0 else None

    def _normalize_live_continuous_tick_rows_for_server(self, rows) -> dict:
        from bloombee.client.live_continuous_batching import (
            ENV_ENABLE_LIVE_CONTINUOUS_BATCHING,
            is_live_continuous_batching_enabled,
        )

        if not is_live_continuous_batching_enabled():
            raise RuntimeError(
                f"{ENV_ENABLE_LIVE_CONTINUOUS_BATCHING}=1 is required to stage live continuous batching rows"
            )
        normalized_rows = list(rows)
        if not normalized_rows:
            raise ValueError("live continuous batching tick rows must be non-empty")
        tick = int(normalized_rows[0].tick)
        if any(int(row.tick) != tick for row in normalized_rows):
            raise ValueError("all live continuous batching rows in one batch must share a tick")
        request_ids = [str(row.request_id) for row in normalized_rows]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("duplicate request_id in live continuous batching tick")
        return {
            "tick": tick,
            "request_ids": request_ids,
            "positions": [int(row.position) for row in normalized_rows],
            "input_token_ids": [int(row.input_token_id) for row in normalized_rows],
        }

    def stage_live_continuous_tick_rows(self, rows) -> dict:
        batch = self._normalize_live_continuous_tick_rows_for_server(rows)
        self._pending_live_continuous_tick_batch = dict(batch)
        return dict(batch)

    def stage_live_continuous_tick_batch(self, batch: dict) -> dict:
        staged = {
            "tick": int(batch["tick"]),
            "request_ids": [str(item) for item in batch["request_ids"]],
            "positions": [int(item) for item in batch["positions"]],
            "input_token_ids": [int(item) for item in batch["input_token_ids"]],
        }
        if "active_mask" in batch and batch["active_mask"] is not None:
            active_mask = [bool(item) for item in batch["active_mask"]]
            if len(active_mask) != len(staged["request_ids"]):
                raise ValueError("active_mask length must match live continuous batching rows")
            staged["active_mask"] = active_mask
        for key in ("batch_offset", "full_batch_size", "micro_batch_size"):
            if key in batch and batch[key] is not None:
                staged[key] = int(batch[key])
        self._pending_live_continuous_tick_batch = staged
        return dict(staged)

    def _consume_live_continuous_batching_metadata(self) -> dict | None:
        if not self._pending_live_continuous_tick_batch:
            return None
        from bloombee.client.live_continuous_batching import (
            ENV_ENABLE_LIVE_CONTINUOUS_BATCHING,
            INFERENCE_SESSION_TICK_ROWS_CLAIM_BOUNDARY,
            is_live_continuous_batching_enabled,
        )

        batch = dict(self._pending_live_continuous_tick_batch)
        self._pending_live_continuous_tick_batch = None
        return {
            "source": "bloombee.client.inference_session",
            "claim_boundary": INFERENCE_SESSION_TICK_ROWS_CLAIM_BOUNDARY,
            "opt_in_flag": ENV_ENABLE_LIVE_CONTINUOUS_BATCHING,
            "opt_in_enabled": is_live_continuous_batching_enabled(),
            "request_count": len(batch.get("request_ids", [])),
            "tick_batches": [batch],
            "live_server_proven": False,
            "speedup_proven": False,
            "wallclock_speedup_proven": False,
            "can_update_demo_status": False,
        }

    def _record_response_metadata(self, response: runtime_pb2.ExpertResponse) -> None:
        if response is None or not response.metadata:
            return
        try:
            metadata = MSGPackSerializer.loads(response.metadata)
        except Exception:
            return
        if isinstance(metadata, dict) and metadata:
            self._server_response_metadata_events.append(metadata)

    def consume_server_response_metadata_events(self) -> list[dict]:
        events = [dict(event) for event in self._server_response_metadata_events]
        self._server_response_metadata_events.clear()
        return events

    def step(
        self,
        inputs: torch.Tensor,
        prompts: torch.Tensor,
        hypo_ids: torch.LongTensor,
        tree_attention_mask: Optional[torch.Tensor] = None,
        kv_cache_position_ids: Optional[torch.Tensor] = None,
        draft_tokens: Optional[torch.Tensor] = None,
        prefill_length: int = 0,
        keep_indices: Optional[torch.Tensor] = None,
        need_pruning: bool = False,
        is_spec_dec: bool = False,
        *,
        step_id: str,
    ) -> torch.Tensor:
        """
        Inference step: send a chunk of input tensors and receive a chunk of outputs
        :prompts: optional DEEP prompts, added to a prefix of each layer's outputs,
          if specified, deep prompts should have shape [num_layers, batch_size, prefix_len, hid_size]
        """
        if self.closed:
            raise Exception("Session is closed, cannot perform step")
        live_tick_batch = self._pending_live_continuous_tick_batch if isinstance(self._pending_live_continuous_tick_batch, dict) else {}
        live_batch_offset = int(live_tick_batch.get("batch_offset", 0) or 0)
        live_full_batch_size = int(
            live_tick_batch.get(
                "full_batch_size",
                self.session_metadata.get("live_continuous_full_batch_size", 0),
            )
            or 0
        )
        live_micro_batch_size = int(live_tick_batch.get("micro_batch_size", inputs.shape[0] if inputs.ndim >= 1 else 1) or 1)
        if is_spec_dec:
            n_input_tokens = 0 if kv_cache_position_ids is None else kv_cache_position_ids[0].numel()
        else:
            n_input_tokens = inputs.shape[1]
        # print('client step() n_input_tokens', n_input_tokens)
        live_microbatch_active = (
            live_full_batch_size > 0
            and inputs.ndim >= 2
            and (live_full_batch_size > inputs.shape[0] or live_batch_offset > 0)
        )
        if live_microbatch_active:
            required_batch = max(live_full_batch_size, live_batch_offset + int(inputs.shape[0]))
            required_len = self._position + n_input_tokens
            if self.history is None:
                self.history = inputs.new_zeros((required_batch, required_len, inputs.shape[2]))
            else:
                history = self.history
                if history.shape[0] < required_batch or history.shape[1] < required_len:
                    expanded = history.new_zeros(
                        (
                            max(history.shape[0], required_batch),
                            max(history.shape[1], required_len),
                            history.shape[2],
                        )
                    )
                    expanded[: history.shape[0], : history.shape[1], :] = history
                    self.history = expanded
            self.history[
                live_batch_offset : live_batch_offset + inputs.shape[0],
                self._position : self._position + n_input_tokens,
                :,
            ] = inputs[:, -n_input_tokens:, :]
        elif self.history is None: # if the history log is empty
            self.history = inputs # assign the current inputs to the history log
        elif self.history.shape[1] == self._position: # if the length of the history equals the current position
            self.history = torch.cat([self.history, inputs[:, -n_input_tokens:]], dim=1) # append the last n_input_tokens of the current input to history
        # history can cat input if it's spec decoding and pruning happened, need fall  back
        # assert self.history.shape[1] == self._position + n_input_tokens,
        #     f"Broken input cache: span={self.span} shape={self.history.shape} "
        #     f"position={self._position} n_input_tokens={n_input_tokens}"
        # )

        if not self.stepped and not live_microbatch_active: # if not exe step yet
            inputs = self.history  # Pass full inputs including prefix
        else:
            inputs = inputs  # No need to pass prefix further
        tokens_to_advance = _server_session_tokens_to_advance(inputs, n_input_tokens, is_spec_dec)

        def _infer_batch_dim(value) -> int:
            if value is None or is_dummy(value):
                return 0
            if torch.is_tensor(value):
                if value.ndim == 0:
                    return 1
                return int(value.shape[0]) if value.shape else 1
            try:
                return int(len(value))
            except Exception:
                return 0

        # For speculative decoding, hidden states may be pruned/compressed on some steps.
        # Derive a stable logical full-batch size from all request tensors and pass it
        # explicitly so server-side KV allocation stays consistent across the session.
        logical_full_batch_size = max(
            _infer_batch_dim(inputs),
            _infer_batch_dim(hypo_ids),
            _infer_batch_dim(keep_indices),
            _infer_batch_dim(prefill_length),
            _infer_batch_dim(draft_tokens),
            _infer_batch_dim(tree_attention_mask),
            int(self.session_metadata.get("live_continuous_full_batch_size", 0) or 0),
            live_full_batch_size,
            1,
        )
        push_only_decode = (
            self.config.use_server_to_server
            and getattr(self.config, "push_only_downstream_decode", False)
            and self.stepped
            and self.span.start > 0
            and not is_spec_dec
        )
        transport_phase = "push_only_decode" if push_only_decode else (
            "spec_decode" if is_spec_dec else ("prefill" if not self.stepped else "decode")
        )

        client_inference_logs_enabled = is_log_channel_enabled("client_inference_logs")
        if client_inference_logs_enabled:
            logger.info(f"_ServerInferenceSession  step id {step_id}")
        if push_only_decode and client_inference_logs_enabled:
            logger.info(
                f"[NETWORK_TX] PUSH_ONLY_WAIT | step_id={step_id} | "
                f"blocks={self.span.start}:{self.span.end} | session_id={self.session_id}"
            )

        total_send_bytes = 0
        serialize_time_ms = 0.0

        with transport_profile_scope() as transport_profile:
            if not push_only_decode:
                # Regular decode does not need speculative-only tensors on the
                # hot path. Keep a compact positional layout and let metadata
                # carry control flags such as is_spec_dec.
                use_compact_decode_layout = not is_spec_dec
                if use_compact_decode_layout:
                    has_prompt_payload = prompts is not None and not is_dummy(prompts)
                    has_hypo_payload = hypo_ids is not None and not is_dummy(hypo_ids)
                    if has_prompt_payload or has_hypo_payload:
                        input_tensors = (
                            inputs,
                            normalize_arg(keep_indices),
                            normalize_arg(prefill_length),
                            prompts,
                            hypo_ids,
                        )
                        tensor_debug_names = (
                            "hidden_states",
                            "keep_indices",
                            "prefill_length",
                            "prompts",
                            "hypo_ids",
                        )
                        regular_layout_name = "decode_compact_v2"
                    else:
                        input_tensors = (
                            inputs,
                            normalize_arg(keep_indices),
                            normalize_arg(prefill_length),
                        )
                        tensor_debug_names = (
                            "hidden_states",
                            "keep_indices",
                            "prefill_length",
                        )
                        regular_layout_name = "decode_minimal_v2"
                else:
                    input_tensors = (
                        inputs,
                        normalize_arg(keep_indices),
                        normalize_arg(tree_attention_mask),
                        normalize_arg(kv_cache_position_ids),
                        normalize_arg(draft_tokens),
                        normalize_arg(prefill_length),
                        prompts,
                        hypo_ids,
                    )
                    tensor_debug_names = (
                        "hidden_states",
                        "keep_indices",
                        "tree_attention_mask",
                        "kv_cache_position_ids",
                        "draft_tokens",
                        "prefill_length",
                        "prompts",
                        "hypo_ids",
                    )
                request_metadata: dict[str, Any] = dict(session_id=self.session_id, step_id=step_id)
                if not self.stepped:
                    request_metadata.update(self.session_metadata)
                # Only send non-default control flags; the server already
                # treats missing values as false/zero.
                if is_spec_dec:
                    request_metadata["is_spec_dec"] = 1
                if need_pruning:
                    request_metadata["need_pruning"] = 1
                request_metadata["full_batch_size"] = int(logical_full_batch_size)
                request_metadata["micro_batch_size"] = int(inputs.shape[0]) if inputs.ndim >= 1 else 1
                request_metadata["inference_layout"] = (
                    regular_layout_name if use_compact_decode_layout else "spec_compact_v1"
                )
                if is_spec_dec:
                    request_metadata["start_from_position"] = self._position + n_input_tokens
                elif self._position is not None:
                    request_metadata["start_from_position"] = self._position
                # Enable server-to-server communication to trigger CROSS_GPU_TRANSFER
                # Speculative decoding keeps strict full-batch semantics; avoid cross-stage push.
                if self.config.use_server_to_server:
                    next_servers = self._collect_next_servers()
                    if next_servers:
                        request_metadata["next_servers"] = next_servers

                live_continuous_metadata = self._consume_live_continuous_batching_metadata()
                if live_continuous_metadata is not None:
                    request_metadata["live_continuous_batching"] = live_continuous_metadata
                    live_batches = live_continuous_metadata.get("tick_batches", [])
                    live_batch = live_batches[0] if live_batches and isinstance(live_batches[0], dict) else {}
                    for key in ("batch_offset", "full_batch_size", "micro_batch_size"):
                        if key in live_batch and live_batch[key] is not None:
                            request_metadata[key] = int(live_batch[key])
                    if "full_batch_size" in request_metadata:
                        request_metadata["live_continuous_full_batch_size"] = int(request_metadata["full_batch_size"])

                # TODO: make possible to use different compression method for different tensors
                server_side_inference_schema, kwargs_schema = self.rpc_info["inference_schema"]
                compression = server_side_inference_schema[0].compression
                server_hidden_states_dtype = _server_hidden_states_wire_dtype(server_side_inference_schema)
                prepared_inputs = [
                    _prepare_rpc_inference_tensor_for_wire(
                        tensor,
                        tensor_debug_names[idx] if idx < len(tensor_debug_names) else f"arg_{idx}",
                        compression,
                        server_hidden_states_dtype,
                    )
                    for idx, tensor in enumerate(input_tensors)
                ]
                hidden_wire_tensor, _, hidden_dtype_debug = prepared_inputs[0]
                capture_wire_activation(
                    hidden_wire_tensor,
                    source="client",
                    channel="rpc_inference",
                    direction="client_to_server",
                    phase=transport_phase,
                    blocks=f"{self.span.start}:{self.span.end}",
                    compute_dtype=str(hidden_dtype_debug.get("compute_dtype", "")),
                    schema_dtype=str(hidden_dtype_debug.get("schema_dtype", "")),
                    wire_dtype=str(hidden_dtype_debug.get("wire_dtype", "")),
                    batch_size=int(logical_full_batch_size),
                    prompt_len=int(hidden_wire_tensor.shape[1]) if hidden_wire_tensor.ndim >= 2 else 1,
                )
                # [NETWORK_TIMING] Measure serialization time
                serialize_start = time.perf_counter()

                # Serialize and send data (debug output removed for performance)
                # Fix for bus error in cross-machine setups: ensure tensors are contiguous before serialization.
                serialized_tensors = [
                    serialize_torch_tensor(
                        wire_tensor,
                        proto.compression,
                        debug_context={
                            "phase": transport_phase,
                            "tensor_name": tensor_debug_names[idx] if idx < len(tensor_debug_names) else f"arg_{idx}",
                            "source": "client",
                            "channel": "rpc_inference",
                            "blocks": f"{self.span.start}:{self.span.end}",
                            "batch": int(logical_full_batch_size),
                            **dtype_debug,
                        },
                    )
                    for idx, (wire_tensor, proto, dtype_debug) in enumerate(prepared_inputs)
                ]
                serialized_metadata = MSGPackSerializer.dumps(request_metadata)

                serialize_end = time.perf_counter()
                serialize_time_ms = (serialize_end - serialize_start) * 1000

                # [NETWORK_TIMING] Measure serialized data size
                total_tensor_bytes = sum(len(t.buffer) for t in serialized_tensors)
                metadata_bytes = len(serialized_metadata)
                total_send_bytes = total_tensor_bytes + metadata_bytes

                if client_inference_logs_enabled:
                    logger.info(f"[NETWORK_TX] SEND_START | step_id={step_id} | "
                               f"tensor_size={total_tensor_bytes/1024:.2f}KB | "
                               f"metadata_size={metadata_bytes}B | "
                               f"total={total_send_bytes/1024:.2f}KB | "
                               f"serialize_time={serialize_time_ms:.2f}ms")

            # [NETWORK_TIMING] Measure network round-trip time
            network_start = time.perf_counter()
            if push_only_decode:
                outputs_serialized = RemoteExpertWorker.run_coroutine(self._await_pushed_step())
            else:
                outputs_serialized = RemoteExpertWorker.run_coroutine(
                    self._step(
                        runtime_pb2.ExpertRequest(
                            uid=self.uid,
                            tensors=serialized_tensors,
                            metadata=serialized_metadata,
                        )
                    )
                )
            self._record_response_metadata(outputs_serialized)

            network_end = time.perf_counter()
            network_rtt_ms = (network_end - network_start) * 1000

            # [NETWORK_TIMING] Measure deserialization time
            deserialize_start = time.perf_counter()
            outputs = list(map(deserialize_torch_tensor, outputs_serialized.tensors))
            deserialize_end = time.perf_counter()
            deserialize_time_ms = (deserialize_end - deserialize_start) * 1000
        
        # [NETWORK_TIMING] Measure received data size
        total_recv_bytes = sum(len(t.buffer) for t in outputs_serialized.tensors)
        
        if client_inference_logs_enabled:
            logger.info(f"[NETWORK_TX] RECV_END | step_id={step_id} | "
                       f"recv_size={total_recv_bytes/1024:.2f}KB | "
                       f"network_rtt={network_rtt_ms:.2f}ms | "
                       f"deserialize_time={deserialize_time_ms:.2f}ms")
        
        # [NETWORK_TIMING] Summary log
        total_time_ms = serialize_time_ms + network_rtt_ms + deserialize_time_ms
        if client_inference_logs_enabled:
            logger.info(f"[NETWORK_TX] SUMMARY | step_id={step_id} | "
                       f"send={total_send_bytes/1024:.2f}KB | recv={total_recv_bytes/1024:.2f}KB | "
                       f"serialize={serialize_time_ms:.2f}ms | network={network_rtt_ms:.2f}ms | "
                       f"deserialize={deserialize_time_ms:.2f}ms | total={total_time_ms:.2f}ms")
        log_transport_profile_event(
            logger,
            source="client",
            channel="rpc_inference",
            blocks=f"{self.span.start}:{self.span.end}",
            step_id=step_id,
            batch_size=int(logical_full_batch_size),
            stats=transport_profile,
            extra={
                "peer": str(self.span.peer_id),
                "phase": transport_phase,
                "seq_tokens": int(inputs.shape[1]) if inputs.ndim >= 2 else 1,
            },
        )
        # assert (
        #     outputs[0].shape == inputs.shape
        # ), f"output activation shape is different from input shape: {outputs[0].shape} != {inputs.shape}"

        self._position += tokens_to_advance
        if client_inference_logs_enabled:
            logger.info(f"server inference session self._position: {self._position}")
        return outputs

    def _collect_next_servers(self) -> List[Tuple[str, str, int, int]]:
        next_servers = []
        session = self.next_session
        while session is not None and session.stepped: 
            next_servers.append(
                (session.span.peer_id.to_base58(), session.session_id, session.span.start, session.span.end)
            )
            session = session.next_session
        return next_servers

    async def _step(self, inputs_serialized: runtime_pb2.ExpertRequest) -> runtime_pb2.ExpertResponse:
        """Inference step on serialized data. This code is meant to be run inside RemoteExpertWorker"""
        await self._inputs_queue.put(inputs_serialized)
        self.stepped = True
        return await asyncio.wait_for(anext(self._outputs_stream), self.config.request_timeout)

    async def _await_pushed_step(self) -> runtime_pb2.ExpertResponse:
        """Wait for the next pushed decode output on an already-open downstream session."""
        return await asyncio.wait_for(anext(self._outputs_stream), self.config.request_timeout)

    def close(self):
        """Finish a given inference session, close the underlying connection"""
        if self._outputs_stream is None:
            return  # already closed
        RemoteExpertWorker.run_coroutine(self._aclose_stream())
        self._outputs_stream = self._inputs_queue = None
        self.closed = True

    async def _aclose_stream(self):
        """Close the inference session. This code is meant to be run inside RemoteExpertWorker"""
        if self._outputs_stream is None:
            return  # already closed
        if self.stepped:
            await self._inputs_queue.put(runtime_pb2.ExpertRequest())  # empty request will trigger end of session
            try:
                await anext(self._outputs_stream)
            except StopAsyncIteration:
                pass

    def __del__(self):
        self.close()

    def __enter__(self):
        assert not self.closed
        return self

    def __exit__(self, *exc_details):
        self.close()


class InferenceSession:
    """
    An interface to a multi-step *inference* session for a sequence of remote transformer blocks
    """

    def __init__(self, sequence_manager: RemoteSequenceManager, max_length: int):
        self._sequence_manager = sequence_manager
        self._closed = False
        self._server_sessions = []
        self._position = 0
        self._max_length = max_length
        self.output_ids = None
        self.past_key_values = None
        self.keep_indices = None
        self.prefill_length = 0
        self._step_count = 0  # Track step count for logging
        self._live_continuous_tick_batches = []
        self._pending_live_continuous_tick_batch = None
        self._live_continuous_full_batch_size = 0
        self._live_continuous_server_observations = []
        self._kv_prefix_reuse_events = []
        self._kv_prefix_reuse_server_observations = []
        
        # [MBPIPE] Log micro-batch pipeline configuration at client session creation
        mbpipe_log_config(logger, context="InferenceSession.__init__")
        self.first_inference = True

    @property
    def num_blocks(self) -> int:
        return len(self._sequence_manager)

    @property
    def position(self) -> int:
        return self._position

    def _normalize_live_continuous_tick_rows(self, rows, *, active_mask=None, output_token_ids=None, output_logits_sha256=None, output_logits_summary=None, output_logits_values=None) -> dict:
        """Normalize opt-in live-continuous decode rows for this session.

        This is deliberately telemetry/control-plane only: staged or recorded
        rows do not prove server-side continuous batching, parity, or speedup.
        It is also fail-closed; callers must be behind
        ``BLOOMBEE_ENABLE_LIVE_CONTINUOUS_BATCHING=1`` rather than silently
        accumulating rows when the experimental path is disabled.
        """
        from bloombee.client.live_continuous_batching import (
            ENV_ENABLE_LIVE_CONTINUOUS_BATCHING,
            is_live_continuous_batching_enabled,
        )

        if not is_live_continuous_batching_enabled():
            raise RuntimeError(
                f"{ENV_ENABLE_LIVE_CONTINUOUS_BATCHING}=1 is required to record live continuous batching rows"
            )

        normalized_rows = list(rows)
        if not normalized_rows:
            raise ValueError("live continuous batching tick rows must be non-empty")

        tick = int(normalized_rows[0].tick)
        if any(int(row.tick) != tick for row in normalized_rows):
            raise ValueError("all live continuous batching rows in one batch must share a tick")
        if self._live_continuous_tick_batches and tick < int(self._live_continuous_tick_batches[-1]["tick"]):
            raise ValueError("live continuous batching ticks must be monotonic")

        request_ids = [str(row.request_id) for row in normalized_rows]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("duplicate request_id in live continuous batching tick")

        batch = {
            "tick": tick,
            "request_ids": request_ids,
            "positions": [int(row.position) for row in normalized_rows],
            "input_token_ids": [int(row.input_token_id) for row in normalized_rows],
        }
        if active_mask is not None:
            normalized_active_mask = [bool(value) for value in active_mask]
            if len(normalized_active_mask) != len(normalized_rows):
                raise ValueError("active_mask length must match live continuous batching rows")
            batch["active_mask"] = normalized_active_mask
        if output_token_ids is not None:
            normalized_outputs = [int(token) for token in output_token_ids]
            if len(normalized_outputs) != len(normalized_rows):
                raise ValueError("output_token_ids length must match live continuous batching rows")
            batch["output_token_ids"] = normalized_outputs
        if output_logits_sha256 is not None:
            normalized_logits_hashes = [str(value) for value in output_logits_sha256]
            if len(normalized_logits_hashes) != len(normalized_rows):
                raise ValueError("output_logits_sha256 length must match live continuous batching rows")
            if any(not value for value in normalized_logits_hashes):
                raise ValueError("output_logits_sha256 values must be non-empty strings")
            batch["output_logits_sha256"] = normalized_logits_hashes
        if output_logits_summary is not None:
            summaries = []
            for item in output_logits_summary:
                if not isinstance(item, dict):
                    raise ValueError("output_logits_summary entries must be objects")
                summaries.append(
                    {
                        "top1_token_id": int(item["top1_token_id"]),
                        "top1_logit": float(item["top1_logit"]),
                        "top2_logit": float(item["top2_logit"]),
                        "top1_margin": float(item["top1_margin"]),
                    }
                )
            if len(summaries) != len(normalized_rows):
                raise ValueError("output_logits_summary length must match live continuous batching rows")
            batch["output_logits_summary"] = summaries
        if output_logits_values is not None:
            values = [[float(value) for value in row] for row in output_logits_values]
            if len(values) != len(normalized_rows):
                raise ValueError("output_logits_values length must match live continuous batching rows")
            vocab_lengths = {len(row) for row in values}
            if 0 in vocab_lengths:
                raise ValueError("output_logits_values rows must be non-empty")
            if len(vocab_lengths) != 1:
                raise ValueError("output_logits_values rows must share a vocab length")
            batch["output_logits_values"] = values
        return batch

    def stage_live_continuous_tick_rows(self, rows) -> dict:
        """Stage a live-continuous batch for the next rpc_inference metadata.

        The server can only observe client batching if the batch metadata is
        present before ``RemoteSequential`` sends hidden states. Generated output
        IDs are still recorded later by ``record_live_continuous_tick_rows``;
        this staged payload is claim-bounded and never proves parity/speedup.
        """

        batch = self._normalize_live_continuous_tick_rows(rows)
        self._pending_live_continuous_tick_batch = dict(batch)
        return dict(batch)

    def stage_live_continuous_tick_batch(self, batch: dict) -> dict:
        staged = {
            "tick": int(batch["tick"]),
            "request_ids": [str(item) for item in batch["request_ids"]],
            "positions": [int(item) for item in batch["positions"]],
            "input_token_ids": [int(item) for item in batch["input_token_ids"]],
        }
        if "active_mask" in batch and batch["active_mask"] is not None:
            active_mask = [bool(item) for item in batch["active_mask"]]
            if len(active_mask) != len(staged["request_ids"]):
                raise ValueError("active_mask length must match live continuous batching rows")
            staged["active_mask"] = active_mask
        for key in ("batch_offset", "full_batch_size", "micro_batch_size"):
            if key in batch and batch[key] is not None:
                staged[key] = int(batch[key])
        self._pending_live_continuous_tick_batch = staged
        return dict(staged)

    def record_live_continuous_tick_rows(
        self,
        rows,
        *,
        active_mask=None,
        output_token_ids=None,
        output_logits_sha256=None,
        output_logits_summary=None,
        output_logits_values=None,
    ) -> None:
        batch = self._normalize_live_continuous_tick_rows(
            rows,
            active_mask=active_mask,
            output_token_ids=output_token_ids,
            output_logits_sha256=output_logits_sha256,
            output_logits_summary=output_logits_summary,
            output_logits_values=output_logits_values,
        )
        self._live_continuous_tick_batches.append(batch)

    def record_server_response_metadata(self, metadata: dict) -> None:
        if not isinstance(metadata, dict):
            return
        live_observed = metadata.get("live_continuous_batching_server_observed")
        if isinstance(live_observed, dict):
            self._live_continuous_server_observations.append(dict(live_observed))
        kv_observed = metadata.get("kv_prefix_reuse_server_observed")
        if isinstance(kv_observed, dict):
            self._kv_prefix_reuse_server_observations.append(dict(kv_observed))

    def live_continuous_batching_report(self) -> dict:
        from bloombee.client.live_continuous_batching import (
            ENV_ENABLE_LIVE_CONTINUOUS_BATCHING,
            INFERENCE_SESSION_TICK_ROWS_CLAIM_BOUNDARY,
            is_live_continuous_batching_enabled,
        )

        request_ids = []
        seen = set()
        for batch in self._live_continuous_tick_batches:
            for request_id in batch["request_ids"]:
                if request_id not in seen:
                    request_ids.append(request_id)
                    seen.add(request_id)

        server_observed_live_continuous_batches = any(
            bool(observation.get("server_observed_live_continuous_batches"))
            for observation in self._live_continuous_server_observations
            if isinstance(observation, dict)
        )
        return {
            "source": "bloombee.client.inference_session",
            "claim_boundary": INFERENCE_SESSION_TICK_ROWS_CLAIM_BOUNDARY,
            "opt_in_flag": ENV_ENABLE_LIVE_CONTINUOUS_BATCHING,
            "opt_in_enabled": is_live_continuous_batching_enabled(),
            "request_count": len(request_ids),
            "total_decode_batches": len(self._live_continuous_tick_batches),
            "tick_batches": [dict(batch) for batch in self._live_continuous_tick_batches],
            "server_observations": [dict(observation) for observation in self._live_continuous_server_observations],
            "server_observed_live_continuous_batches": server_observed_live_continuous_batches,
            "live_server_proven": bool(self._live_continuous_server_observations),
            "speedup_proven": False,
            "can_update_demo_status": False,
        }

    def record_kv_prefix_reuse_prefill(self, input_ids, *, request_ids=None) -> dict:
        """Record same-prefix/varied-suffix prefill metadata behind opt-in flag.

        This is metadata only: it proves the session can identify reusable
        prefix structure, not that server-side KV slabs were reused.
        """
        from bloombee.client.kv_prefix_reuse import (
            ENV_ENABLE_KV_PREFIX_REUSE,
            build_prefill_metadata_event,
            is_kv_prefix_reuse_enabled,
        )

        if not is_kv_prefix_reuse_enabled():
            raise RuntimeError(
                f"{ENV_ENABLE_KV_PREFIX_REUSE}=1 is required to record KV prefix reuse metadata"
            )
        event = build_prefill_metadata_event(input_ids, request_ids=request_ids)
        event["opt_in_enabled"] = True
        self._kv_prefix_reuse_events.append(event)
        metadata = getattr(self, "session_metadata", None)
        if isinstance(metadata, dict):
            metadata["kv_prefix_reuse"] = self.kv_prefix_reuse_report()
        return event

    def kv_prefix_reuse_report(self) -> dict:
        from bloombee.client.kv_prefix_reuse import (
            CLAIM_BOUNDARY,
            ENV_ENABLE_KV_PREFIX_REUSE,
            is_kv_prefix_reuse_enabled,
        )

        server_observed_kv_cache_reuse = any(
            bool(observation.get("server_observed_kv_cache_reuse"))
            for observation in self._kv_prefix_reuse_server_observations
            if isinstance(observation, dict)
        )
        return {
            "source": "bloombee.client.inference_session",
            "claim_boundary": CLAIM_BOUNDARY,
            "opt_in_flag": ENV_ENABLE_KV_PREFIX_REUSE,
            "opt_in_enabled": is_kv_prefix_reuse_enabled(),
            "event_count": len(self._kv_prefix_reuse_events),
            "events": [dict(event) for event in self._kv_prefix_reuse_events],
            "server_observations": [dict(observation) for observation in self._kv_prefix_reuse_server_observations],
            "runtime_prefill_metadata_proven": bool(self._kv_prefix_reuse_events),
            "server_observed_kv_cache_reuse": server_observed_kv_cache_reuse,
            "live_kv_cache_reuse_proven": server_observed_kv_cache_reuse,
            "speedup_proven": False,
            "can_update_demo_status": False,
        }

    @position.setter
    def position(self, start_from_position: int) -> None:
        # Set the position and keep all related session objects in sync.
        self._position = start_from_position
        for session in self._server_sessions:
            assert isinstance(session, _ServerInferenceSession)
            session.position = start_from_position

    def _enter_server_sessions(self, chosen_spans: List[RemoteSpanInfo]) -> List[_ServerInferenceSession]:
        server_sessions = []  # build server sessions; on error, ensure already-created ones exit cleanly
        try:
            for span in chosen_spans:
                span_uids = CHAIN_DELIMITER.join(self._sequence_manager.block_uids[span.start : span.end])
                metadata = self._sequence_manager.get_request_metadata(
                    "rpc_inference", span_uids, peer_id=span.peer_id
                ) or {}
                kv_prefix_report = self.kv_prefix_reuse_report()
                live_full_batch_size = int(getattr(self, "_live_continuous_full_batch_size", 0) or 0)
                if live_full_batch_size > 0:
                    metadata["live_continuous_full_batch_size"] = live_full_batch_size
                if (
                    kv_prefix_report.get("opt_in_enabled")
                    and kv_prefix_report.get("runtime_prefill_metadata_proven")
                ):
                    metadata["kv_prefix_reuse"] = kv_prefix_report
                session = RemoteExpertWorker.run_coroutine(
                    _ServerInferenceSession.create(
                        self._sequence_manager.config,
                        self._sequence_manager.state.p2p,
                        span,
                        span_uids,
                        rpc_info=self._sequence_manager.rpc_info,
                        max_length=self._max_length,
                        **metadata,
                    )
                )
                server_sessions.append(session)
                session.__enter__()
            return server_sessions
        except Exception:
            self._exit_server_sessions(server_sessions)
            raise

    def _exit_server_sessions(self, server_sessions: List[_ServerInferenceSession]) -> None:
        for session in reversed(server_sessions):
            try:
                session.__exit__(None, None, None)
            except Exception:
                logger.debug("Caught exception while closing connection to server:", exc_info=True)

    def __enter__(self) -> "InferenceSession":
        assert not self._closed and not self._server_sessions
        return self

    # Execute one inference step over inputs / prompts / hypothesis IDs,
    # retrying on transient errors.
    def step(
        self,
        inputs: torch.Tensor,
        prompts: Optional[torch.Tensor] = None,
        hypo_ids: Optional[torch.Tensor] = None,
        tree_attention_mask: Optional[torch.Tensor] = None,
        kv_cache_position_ids: Optional[torch.Tensor] = None,
        draft_tokens: Optional[torch.Tensor] = None,
        is_spec_decoding: Optional[torch.Tensor] = None,
        prefill_length: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert not self._closed
        if torch.is_grad_enabled():
            logger.warning("Running inference session with grad enabled. Gradients will *not* be propagated correctly.")

        if prompts is None or is_dummy(prompts):
            prompts = DUMMY
        else:
            assert prompts.ndim == 4, "deep prompts should have shape [num_blocks, batch_size, prefix_len, hid_size]"
            assert prompts.shape[0] == self.num_blocks
            assert prompts.shape[1] in (inputs.shape[0], 1)
            assert prompts.shape[2] <= inputs.shape[1]
            assert prompts.shape[3] == inputs.shape[2]

        if hypo_ids is None or is_dummy(hypo_ids):
            hypo_ids = DUMMY_INT64
        else:
            assert len(hypo_ids) == len(inputs)
            assert hypo_ids.dtype == torch.int64

        inputs_device = inputs.device
        inputs_dtype = inputs.dtype
        
        inputs = inputs.cpu()
        prompts = prompts.cpu()
        hypo_ids = hypo_ids.cpu()
        tree_attention_mask = tree_attention_mask.cpu() if tree_attention_mask is not None else None
        kv_cache_position_ids = kv_cache_position_ids.cpu() if kv_cache_position_ids is not None else None
        draft_tokens = draft_tokens.cpu() if draft_tokens is not None else None
        is_spec_decoding = is_spec_decoding.cpu() if is_spec_decoding is not None else None
        
        step_id = str(uuid.uuid4())  # Generate a unique step ID.
        
        # [MBPIPE] Log current path at client step entry (first step only to reduce noise)
        self._step_count += 1
        if self._step_count == 1:
            batch_size = inputs.shape[0] if inputs.ndim >= 1 else 1
            mbpipe_log_path_entry(logger, "client.InferenceSession.step", batch_size=batch_size)

        n_input_tokens = inputs.shape[1] if kv_cache_position_ids is None else kv_cache_position_ids[0].numel()
        if self._position + n_input_tokens > self._max_length:
            raise ValueError(
                f"Maximum length exceeded: prefix {self._position} + current {n_input_tokens} exceeds pre-allocated maximum {self._max_length}"
            )

        server_idx = 0
        block_idx = 0
        inference_step_start = time.perf_counter()
        batch_size = inputs.shape[0] if inputs.ndim >= 1 else 1
        if prefill_length is not None:
            self.prefill_length = prefill_length.to(inputs.device)
        else:
            self.prefill_length = torch.zeros(batch_size, device=inputs.device)
        keep_indices = torch.arange(
            inputs.shape[1],
            dtype=torch.int64,
            device=inputs.device
        ).unsqueeze(0).expand(inputs.shape[0], -1)
        self.keep_indices = keep_indices
        if torch.is_tensor(is_spec_decoding):
            is_spec_dec = bool(is_spec_decoding.detach().bool().any().item()) if is_spec_decoding.numel() > 0 else False
        else:
            is_spec_dec = bool(is_spec_decoding)
        need_pruning = is_spec_dec
        while block_idx < self.num_blocks:
            for attempt_no in itertools.count():
                logger.debug(f"Inference: block {block_idx}, attempt {attempt_no}")
                server_session = None
                try:
                    if not self._server_sessions or attempt_no >= 1:
                        self._update_sequence(server_idx, block_idx, attempt_no)

                    server_session = self._server_sessions[server_idx]
                    # assert server_session.position == self.position, f"{server_session.position} and {self.position}"
                    
                    # 🔍 CLIENT DEBUG: Log server span processing start
                    span_start_time = time.perf_counter()
                    
                    server_inputs = _trim_recovered_history_for_existing_downstream(
                        inputs,
                        current_step_tokens=n_input_tokens,
                        downstream_position=server_session.position,
                        is_spec_dec=is_spec_dec,
                    )
                    if server_idx == 0 and self._pending_live_continuous_tick_batch is not None:
                        server_session.stage_live_continuous_tick_batch(self._pending_live_continuous_tick_batch)
                        self._pending_live_continuous_tick_batch = None
                    inputs, keep_indices, *_ = server_session.step(
                        server_inputs,
                        prompts[server_session.span.start : server_session.span.end],
                        hypo_ids,
                        tree_attention_mask,
                        kv_cache_position_ids,
                        draft_tokens,
                        self.prefill_length,
                        self.keep_indices,
                        need_pruning,
                        is_spec_dec,
                        step_id=step_id,
                    )
                    for response_metadata in server_session.consume_server_response_metadata_events():
                        self.record_server_response_metadata(response_metadata)
                    if is_spec_dec and need_pruning:
                        self.keep_indices = keep_indices
                    
                    need_pruning = False  # only need to prune on the first server
                    
                    # 🔍 CLIENT DEBUG: Log server span processing end
                    span_end_time = time.perf_counter()
                    span_duration = (span_end_time - span_start_time) * 1000  # ms
                    if is_log_channel_enabled("client_inference_logs"):
                        logger.info(
                            f"[CLIENT_SERVER_END] ServerIdx={server_idx} | Blocks={server_session.span.start}:{server_session.span.end} | Duration={span_duration:.2f}ms"
                        )
                    # print('inputs ', inputs)
                    # print('inputs.shape ', inputs.shape)
                    server_idx += 1
                    block_idx = server_session.span.end
                    self._sequence_manager.on_request_success(server_session.span.peer_id)
                    break
                except Exception as e:
                    self._sequence_manager.on_request_failure(
                        server_session.span.peer_id if server_session is not None else None
                    )
                    if attempt_no + 1 == self._sequence_manager.config.max_retries:
                        raise
                    delay = self._sequence_manager.get_retry_delay(attempt_no)
                    logger.warning(
                        _format_recovery_retry_event(
                            span=server_session.span if server_session is not None else None,
                            attempt_no=attempt_no,
                            max_retries=self._sequence_manager.config.max_retries,
                            delay_s=delay,
                            error=e,
                        )
                    )
                    maybe_log_traceback(e)
                    time.sleep(delay) 

        self._position += n_input_tokens
        # logger.info(f"keep_indices: {keep_indices}")
        # logger.info(f"before _recover_hidden_states: {inputs}")
        # t0 = time.perf_counter()
        if draft_tokens is not None and is_spec_dec:
            inputs = self._restore_hidden_states(inputs, self.keep_indices, draft_tokens.shape[1])
        # t1 = time.perf_counter()
        # logger.info(f"_restore_hidden_states took {(t1 - t0) * 1000:.2f} ms")
        # logger.info(f"after _recover_hidden_states: {inputs}")
        outputs = inputs
        # A retried downstream server session may resend full history to rebuild its
        # server-side cache, which means the final stage can legitimately return
        # hidden states for the whole cached prefix instead of only the current
        # step's token(s). Regular decode expects only the newly-advanced token
        # window here, but speculative verification needs the full per-tree output
        # tensor, so do not trim speculative steps back to the committed token count.
        if (
            not is_spec_dec
            and torch.is_tensor(outputs)
            and outputs.ndim == 3
            and n_input_tokens > 0
            and outputs.shape[1] > n_input_tokens
        ):
            logger.warning(
                _format_final_history_trim_event(
                    seq_len=int(outputs.shape[1]),
                    current_step_tokens=int(n_input_tokens),
                    client_position=int(self._position),
                )
            )
            outputs = outputs[:, -n_input_tokens:, :]
        elif (
            torch.is_tensor(outputs)
            and outputs.ndim == 3
            and n_input_tokens > 0
            and outputs.shape[1] < n_input_tokens
        ):
            raise RuntimeError(
                "Final stage returned fewer tokens than requested for the current step: "
                f"outputs.shape={tuple(outputs.shape)}, current_step_tokens={n_input_tokens}"
            )

        # 🔍 CLIENT DEBUG: Log inference step end
        inference_step_end = time.perf_counter()
        inference_step_duration = (inference_step_end - inference_step_start) * 1000  # ms
        if is_log_channel_enabled("client_inference_logs"):
            logger.info(
                f"[CLIENT_INFERENCE_END] Position={self._position} | Duration={inference_step_duration:.2f}ms | Servers={server_idx}"
            )
            logger.info("=" * 80)
        
        outputs = outputs.to(device=inputs_device, dtype=inputs_dtype) 
        # print('client inference session outputs ', outputs.shape)
        return outputs
    
    def _restore_hidden_states(
        self,
        flattened_hidden_states: torch.Tensor,
        keep_indices: torch.Tensor,
        original_seq_len: int,
    ) -> torch.Tensor:
        """
        Restore flattened hidden states to [B, original_seq_len, hidden_size].

        Args:
            flattened_hidden_states: [N_total_valid, hidden_size] flattened valid hidden states
            keep_indices: [B, max_keep_len] per-batch keep indices, padded with -1
            original_seq_len: original sequence length

        Returns:
            restored_hidden_states: [B, original_seq_len, hidden_size], invalid positions filled with 0
        """
        batch_size, max_keep_len = keep_indices.shape
        device = flattened_hidden_states.device
        dtype = flattened_hidden_states.dtype
        
        def _flatten_hidden_with_keep_layout(hidden_states: torch.Tensor) -> torch.Tensor:
            if hidden_states.ndim == 2:
                return hidden_states

            if hidden_states.ndim != 3:
                raise ValueError(f"Unexpected flattened_hidden_states dim: {hidden_states.ndim}")

            if tuple(hidden_states.shape[:2]) == tuple(keep_indices.shape):
                valid_mask_local = keep_indices >= 0
                return hidden_states[valid_mask_local]

            num_groups, _, local_hidden_size = hidden_states.shape
            total_batch = int(keep_indices.shape[0])

            if num_groups > 0 and total_batch % num_groups == 0:
                batch_per_group = total_batch // num_groups
                grouped_rows = []
                for group_idx in range(num_groups):
                    keep_chunk = keep_indices[
                        group_idx * batch_per_group : (group_idx + 1) * batch_per_group
                    ]
                    valid_count = int((keep_chunk >= 0).sum().item())
                    if valid_count == 0:
                        continue

                    group_hidden = hidden_states[group_idx]
                    if int(group_hidden.shape[0]) < valid_count:
                        raise ValueError(
                            f"Spec micro-batch hidden rows are shorter than valid keep entries: "
                            f"group={group_idx}, hidden_rows={group_hidden.shape[0]}, valid_keep={valid_count}"
                        )
                    grouped_rows.append(group_hidden[:valid_count])

                if grouped_rows:
                    return torch.cat(grouped_rows, dim=0)
                return hidden_states.new_empty((0, local_hidden_size))

            flat_hidden_local = hidden_states.reshape(-1, local_hidden_size)
            expected_valid = int((keep_indices >= 0).sum().item())
            if flat_hidden_local.shape[0] > expected_valid:
                trailing = flat_hidden_local[expected_valid:]
                if trailing.numel() == 0 or not torch.count_nonzero(trailing).item():
                    flat_hidden_local = flat_hidden_local[:expected_valid]
            return flat_hidden_local

        # Handle different input dimensions
        if flattened_hidden_states.ndim == 2:
            # [N_total_valid, hidden_size] -> use directly
            flat_hidden = flattened_hidden_states
            hidden_size = flattened_hidden_states.shape[-1]
        elif flattened_hidden_states.ndim == 3:
            hidden_size = flattened_hidden_states.shape[-1]
            flat_hidden = _flatten_hidden_with_keep_layout(flattened_hidden_states)
        else:
            raise ValueError(f"Unexpected flattened_hidden_states dim: {flattened_hidden_states.ndim}")
        
        # Build output tensor, zero-filled
        restored_hidden_states = torch.zeros(
            batch_size, original_seq_len, hidden_size,
            dtype=dtype, device=device
        )

        # Valid mask: [B, max_keep_len]
        valid_mask = keep_indices >= 0

        # Batch index broadcast: [B, max_keep_len]
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(keep_indices)

        # Extract valid index pairs
        valid_batch_idx = batch_idx[valid_mask]      # [N_total_valid]
        valid_seq_idx = keep_indices[valid_mask]     # [N_total_valid]

        # Verify dimensions match
        n_total_valid = valid_mask.sum().item()
        if flat_hidden.shape[0] != n_total_valid:
            raise ValueError(
                f"Dimension mismatch: flattened_hidden_states has {flat_hidden.shape[0]} elements, "
                f"but keep_indices has {n_total_valid} valid entries"
            )

        # Scatter the valid rows back into their original positions
        restored_hidden_states[valid_batch_idx, valid_seq_idx, :] = flat_hidden
        
        return restored_hidden_states
    
    def _update_sequence(self, server_idx: int, block_idx: int, attempt_no: int) -> int:
        # If there is a failed server session, this code closes it
        self._exit_server_sessions(self._server_sessions[server_idx : server_idx + 1])

        n_prev_spans = len(self._server_sessions)
        update_end = self._server_sessions[server_idx].span.end if server_idx < n_prev_spans else self.num_blocks
        if attempt_no >= 1: 
            logger.debug(
                f"Due to a server failure, remote attention caches "
                f"from block {block_idx} to {update_end} will be regenerated"
            )

        updated_spans = self._sequence_manager.make_sequence(
            block_idx, update_end, mode="min_latency", cache_tokens_needed=self._max_length
        )

        # make_sequence() could return a longer sequence
        updated_spans[-1].end = min(updated_spans[-1].end, update_end)
        updated_sessions = self._enter_server_sessions(updated_spans)
        logger.debug(f"Found path from block {block_idx} to {update_end} via {len(updated_spans)} servers")
        
        
        # If there is a failed span, this code replaces it, otherwise it just adds new ones
        if server_idx < n_prev_spans:
            updated_sessions[0].history = self._server_sessions[server_idx].history
        self._server_sessions[server_idx : server_idx + 1] = updated_sessions

        # Update links to the next server session for direct server-to-server communication via rpc_push()
        for i in range(max(server_idx - 1, 0), min(server_idx + len(updated_spans), len(self._server_sessions) - 1)):
            self._server_sessions[i].next_session = self._server_sessions[i + 1]

    def close(self, *exc_details):
        """Finish a given inference session, close the underlying connection"""
        if not self._closed:
            self._exit_server_sessions(self._server_sessions)
            self._server_sessions.clear()
            self._closed = True

    def __exit__(self, *exc_details):
        self.close(*exc_details)

    def __del__(self):
        self.close()

    @property
    def last_token_id(self) -> Optional[torch.Tensor]:  # Backward compatibility with Petals < 2.1.0
        return self.output_ids[:, -1:] if self.output_ids is not None else None

    @last_token_id.setter
    def last_token_id(self, value: torch.Tensor):  # Backward compatibility with Petals < 2.1.0
        if self.output_ids is None:
            raise RuntimeError("Can't override `last_token_id` since the session has not stepped yet")
        self.output_ids[:, -1:] = value
