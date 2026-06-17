"""Forward hooks that record per-module timing and shape info during inference."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
import torch.nn as nn


class Phase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    UNKNOWN = "unknown"


@dataclass
class ModuleEvent:
    """A single recorded forward pass through one module."""

    module_path: str           # e.g. "model.layers.0.self_attn.q_proj"
    module_type: str           # e.g. "Linear"
    input_shape: tuple | None  # shape of the first tensor input, if any
    output_shape: tuple | None # shape of the output tensor, if any
    duration_ms: float         # how long the forward pass took, in milliseconds
    device: str                # "cuda", "mps", or "cpu"
    phase: Phase = Phase.UNKNOWN


@dataclass
class ModuleTracer:
    """Registers forward hooks on a model and collects per-module timing.

    Uses CUDA events for accurate GPU timing when available, falls back to
    wall-clock time otherwise. CPU/MPS wall-clock timing is approximate but
    fine for development."""

    target_types: tuple[type, ...] = (nn.Linear, nn.LayerNorm)
    events: list[ModuleEvent] = field(default_factory=list)
    memory_samples: list[dict] = field(default_factory=list)
    _handles: list[Any] = field(default_factory=list)
    _pending: dict[int, dict[str, Any]] = field(default_factory=dict)
    _pass_count: int = 0  #tracks forward pass number


    def attach(self, model: nn.Module) -> None:
        """Walk the model and register hooks on every module of a target type."""
        for name, module in model.named_modules():
            if isinstance(module, self.target_types):
                pre_handle = module.register_forward_pre_hook(
                    self._make_pre_hook(name, type(module).__name__)
                )
                post_handle = module.register_forward_hook(
                    self._make_post_hook(name, type(module).__name__)
                )
                self._handles.extend([pre_handle, post_handle])


        self._attach_memory_sampler(model)

    def _attach_memory_sampler(self, model: nn.Module) -> None:
        """Sample GPU memory allocated at the start of each forward pass."""
        import torch
        if not torch.cuda.is_available():
            return

        tracer_ref = self  # capture self for the closure

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                def memory_hook(mod, inputs):
                    mem_bytes = torch.cuda.memory_allocated()
                    phase = tracer_ref._detect_phase(
                        tracer_ref._shape_of(inputs[0]) if inputs else None
                    )
                    tracer_ref.memory_samples.append({
                        "pass": tracer_ref._pass_count,
                        "phase": phase.value,
                        "memory_bytes": mem_bytes,
                    })
                    tracer_ref._pass_count += 1

                handle = module.register_forward_pre_hook(memory_hook)
                self._handles.append(handle)
                break  # first Linear only

        return

    def detach(self) -> None:
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._pending.clear()

    def _make_pre_hook(self, module_path: str, module_type: str):
        def pre_hook(module: nn.Module, inputs: tuple) -> None:
            device = self._device_of(inputs, module)
            input_shape = self._shape_of(inputs[0]) if inputs else None

            timing: dict[str, Any] = {
                "module_path": module_path,
                "module_type": module_type,
                "input_shape": input_shape,
                "device": device,
            }

            if device == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                timing["cuda_start"] = start
                timing["cuda_end"] = end
            else:
                timing["wall_start"] = time.perf_counter()

            self._pending[id(module)] = timing

        return pre_hook

    def _make_post_hook(self, module_path: str, module_type: str):
        def post_hook(module: nn.Module, inputs: tuple, output: Any) -> None:
            timing = self._pending.pop(id(module), None)
            if timing is None:
                return

            output_shape = self._shape_of(output)

            if timing["device"] == "cuda":
                timing["cuda_end"].record()
                torch.cuda.synchronize()
                duration_ms = timing["cuda_start"].elapsed_time(timing["cuda_end"])
            else:
                duration_ms = (time.perf_counter() - timing["wall_start"]) * 1000.0

            # Detect phase from input sequence dimension
            phase = self._detect_phase(timing["input_shape"])

            self.events.append(
                ModuleEvent(
                    module_path=timing["module_path"],
                    module_type=timing["module_type"],
                    input_shape=timing["input_shape"],
                    output_shape=output_shape,
                    duration_ms=duration_ms,
                    device=timing["device"],
                    phase=phase,
                )
            )

        return post_hook

    @staticmethod
    def _shape_of(x: Any) -> tuple | None:
        if isinstance(x, torch.Tensor):
            return tuple(x.shape)
        if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], torch.Tensor):
            return tuple(x[0].shape)
        return None

    @staticmethod
    def _device_of(inputs: tuple, module: nn.Module) -> str:
        # Prefer the input's device; fall back to a parameter's device.
        if inputs and isinstance(inputs[0], torch.Tensor):
            return inputs[0].device.type
        for p in module.parameters():
            return p.device.type
        return "cpu"


    @staticmethod
    def _detect_phase(input_shape: tuple | None) -> Phase:
        """Infer prefill vs decode from the sequence dimension of the input.

        For decoder-only transformers: seq_len > 1 means prefill (processing
        the full prompt). seq_len == 1 means decode (one new token per pass).
        """
        if input_shape is None:
            return Phase.UNKNOWN
        # Shape is (batch, seq_len, hidden_dim) for most transformer layers
        if len(input_shape) >= 2:
            seq_len = input_shape[1]
            if seq_len == 1:
                return Phase.DECODE
            if seq_len > 1:
                return Phase.PREFILL
        return Phase.UNKNOWN
