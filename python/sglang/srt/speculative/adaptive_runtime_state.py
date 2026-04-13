import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from sglang.srt.speculative.adaptive_spec_params import (
    AdaptiveSpeculativeParams,
    load_adaptive_config,
)

logger = logging.getLogger(__name__)


@dataclass
class SpecRuntimeState:
    """A complete set of runtime resources bound to a specific speculative
    decoding configuration.

    Each decode round runs three stages — draft, verify, extend — and every
    stage has shape-dependent resources (attention backends and CUDA graphs)
    that must match the current configuration.  Switching adaptive steps
    means swapping the entire state atomically.
    """

    # -- Configuration (determines shapes for all stages) --
    speculative_num_steps: int
    speculative_num_draft_tokens: int

    # -- Draft stage: draft model multi-step autoregressive generation --
    draft_attn_backend: Optional[object]
    cuda_graph_runner: Optional[object]

    # -- Verify stage: target model one-pass tree verification --
    target_attn_backend: object
    target_graph_runner: Optional[object]

    # -- Extend stage: draft model KV cache catch-up after verify --
    draft_extend_attn_backend: Optional[object]
    cuda_graph_runner_for_draft_extend: Optional[object]


class AdaptiveSpecWorker(Protocol):
    """Protocol that a worker must implement to use AdaptiveController."""

    speculative_num_steps: int

    def build_adaptive_runtime_state(
        self, speculative_num_steps: int, speculative_num_draft_tokens: int
    ) -> Any: ...

    def apply_runtime_state(self, state: Any) -> None: ...


class AdaptiveController:
    """Facade that owns adaptive decision-making and runtime state switching.

    Works with any worker that implements ``AdaptiveSpecWorker`` protocol:
      - ``build_adaptive_runtime_state(steps, draft_tokens)`` → state object
      - ``apply_runtime_state(state)`` → apply it to the worker

    The worker only needs to:
      1. Call ``register()`` for the initial state, then ``init_states()``
         once during startup.
      2. Call ``on_verify_complete(accept_lengths)`` after each decode verify.
    """

    def __init__(self, worker: AdaptiveSpecWorker, config_path: Optional[str] = None):
        self.worker = worker
        cfg = load_adaptive_config(config_path)
        self.params = AdaptiveSpeculativeParams(
            initial_steps=worker.speculative_num_steps,
            config=cfg,
        )
        self._states: Dict[int, Any] = {}

    @property
    def candidate_steps(self):
        return self.params.candidate_steps

    def register(self, state: Any, steps: Optional[int] = None):
        """Register a pre-built runtime state.

        *steps* defaults to ``state.speculative_num_steps`` when not given.
        """
        key = steps if steps is not None else state.speculative_num_steps
        self._states[key] = state

    def init_states(self):
        """Build and register runtime states for all candidate steps."""
        for steps in self.params.candidate_steps:
            if steps in self._states:
                continue
            state = self.worker.build_adaptive_runtime_state(
                speculative_num_steps=steps,
                speculative_num_draft_tokens=steps + 1,
            )
            self._states[steps] = state
        self._activate(self.params.current_steps)

    def on_verify_complete(self, accept_lengths: List[int]) -> None:
        """Feed verify results; switch runtime state if EMA warrants it."""
        if self.params.update(accept_lengths):
            self._activate(self.params.current_steps)

    def _activate(self, speculative_num_steps: int):
        state = self._states.get(speculative_num_steps)
        if state is None:
            raise ValueError(
                f"Missing adaptive runtime state for steps={speculative_num_steps}"
            )
        self.worker.apply_runtime_state(state)
