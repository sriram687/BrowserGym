"""
BrowserGym Experiment Callback System
======================================

Provides a lightweight, extensible, and fully backward-compatible event hook system
for ``ExpArgs.run()``. Callbacks receive live lifecycle events from the experiment
loop without requiring any modifications to existing research scripts.

Usage example::

    from browsergym.experiments.callbacks import ConsoleProgressCallback, BudgetLimitCallback

    exp_args = ExpArgs(
        agent_args=...,
        env_args=...,
        callbacks=[
            ConsoleProgressCallback(),
            BudgetLimitCallback(max_tokens=50_000),
        ],
    )
    exp_args.run()

Callback lifecycle order per episode
--------------------------------------
::

    on_exp_start(exp_args)
        └─ [for each step]
               on_step_start(step)
               on_action_selected(step, action, agent_info)
               on_step_end(step_info, is_done)
    on_exp_end(exp_dir, err_msg)

"""

from __future__ import annotations

import logging
import time
from abc import ABC
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    # Avoid circular imports at runtime; only used for type-hints.
    from browsergym.experiments.loop import ExpArgs, StepInfo

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────


class AbstractCallback(ABC):
    """Base class for experiment lifecycle callbacks.

    All hook methods have a no-op default implementation so that subclasses only
    need to override the events they care about.  This makes it trivial to add
    new lifecycle hooks in the future without breaking existing callback
    implementations.
    """

    def on_exp_start(self, exp_args: "ExpArgs") -> None:
        """Called once, immediately before the agent and environment are created.

        Args:
            exp_args: The full ``ExpArgs`` instance, providing task name, max
                steps, agent name, experiment directory, etc.
        """

    def on_step_start(self, step: int) -> None:
        """Called at the beginning of every agent step (before the agent thinks).

        Args:
            step: Zero-based step index within the current episode.
        """

    def on_action_selected(self, step: int, action: Optional[str], agent_info: dict) -> None:
        """Called immediately after the agent returns an action.

        The environment has *not* yet executed the action.  This is the right
        moment to inspect the raw LLM output, token counts, or agent reasoning.

        Args:
            step: Zero-based step index.
            action: The raw action string returned by the agent, or ``None`` if
                the agent signalled end-of-episode.
            agent_info: The agent's auxiliary information dictionary (may contain
                ``chat_messages``, ``stats``, etc.).
        """

    def on_step_end(self, step_info: "StepInfo", is_done: bool) -> None:
        """Called after the environment has executed the action and ``step_info``
        is fully populated.

        This is the right moment to read reward, token stats, step latency, and
        decide whether to abort the episode.

        Args:
            step_info: The fully-populated ``StepInfo`` for the completed step.
            is_done: ``True`` if this step terminated or truncated the episode.
        """

    def on_exp_end(self, exp_dir, err_msg: Optional[str] = None) -> None:
        """Called once in the ``finally`` block after the episode finishes
        (whether normally, via truncation, or due to an error).

        Args:
            exp_dir: ``pathlib.Path`` pointing at the experiment output directory.
            err_msg: Non-``None`` if the episode ended due to an unhandled exception.
        """


# ─────────────────────────────────────────────────────────────────────────────
# CallbackList — internal dispatcher
# ─────────────────────────────────────────────────────────────────────────────


class CallbackList:
    """Fan-out dispatcher that broadcasts every lifecycle event to a list of
    :class:`AbstractCallback` instances in registration order.

    **Fail-safe**: exceptions raised inside any individual callback are caught,
    logged as a warning, and execution continues.  A misbehaving monitoring
    callback will never crash or corrupt a running benchmark.
    """

    def __init__(self, callbacks: list[AbstractCallback]) -> None:
        self._callbacks: list[AbstractCallback] = list(callbacks)

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _fire(self, method_name: str, *args, **kwargs) -> None:
        """Call ``method_name`` on every registered callback, swallowing errors."""
        for cb in self._callbacks:
            try:
                getattr(cb, method_name)(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Callback %s.%s raised an exception (ignored): %s",
                    type(cb).__name__,
                    method_name,
                    exc,
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    # Public broadcast methods (mirror AbstractCallback's interface)
    # ------------------------------------------------------------------

    def on_exp_start(self, exp_args: "ExpArgs") -> None:
        self._fire("on_exp_start", exp_args)

    def on_step_start(self, step: int) -> None:
        self._fire("on_step_start", step)

    def on_action_selected(self, step: int, action: Optional[str], agent_info: dict) -> None:
        self._fire("on_action_selected", step, action, agent_info)

    def on_step_end(self, step_info: "StepInfo", is_done: bool) -> None:
        self._fire("on_step_end", step_info, is_done)

    def on_exp_end(self, exp_dir, err_msg: Optional[str] = None) -> None:
        self._fire("on_exp_end", exp_dir, err_msg)


# ─────────────────────────────────────────────────────────────────────────────
# Built-in callback: ConsoleProgressCallback
# ─────────────────────────────────────────────────────────────────────────────


class ConsoleProgressCallback(AbstractCallback):
    """Renders a clean, single-line progress display in the terminal.

    Instead of scrolling through raw debug logs, researchers see a compact,
    human-readable line that is rewritten in place after every step::

        [MyAgent @ webarena.241]  step 4/30 | reward 0.00 | tokens  1,842 | step  2.3s | elapsed  9.4s

    Features
    --------
    * **Single-line rewrite**: Uses ``\\r`` so the line is updated in-place.
      Falls back to plain newlines when the output stream is not a TTY (e.g.
      redirected to a file or CI log).
    * **Zero dependencies**: Uses only Python's ``time`` and ``sys`` stdlib.
    * **Graceful finish**: Prints a summary line when the episode ends.

    Args:
        show_action_preview: When ``True``, prints a short preview (first
            ``action_preview_chars`` characters) of the selected action on its
            own line.  Defaults to ``False`` to keep output compact.
        action_preview_chars: Maximum number of characters to show in the action
            preview.  Defaults to 120.
    """

    def __init__(
        self,
        show_action_preview: bool = False,
        action_preview_chars: int = 120,
    ) -> None:
        self._show_action_preview = show_action_preview
        self._action_preview_chars = action_preview_chars

        # Internal state reset per episode.
        self._exp_start_time: float = 0.0
        self._step_start_time: float = 0.0
        self._agent_name: str = "agent"
        self._task_name: str = "task"
        self._max_steps: int = 0
        self._cumulative_tokens: int = 0
        self._cumulative_reward: float = 0.0
        self._is_tty: bool = False

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_exp_start(self, exp_args: "ExpArgs") -> None:
        import sys

        self._exp_start_time = time.monotonic()
        self._agent_name = getattr(exp_args.agent_args, "agent_name", "agent") or "agent"
        self._task_name = exp_args.env_args.task_name
        self._max_steps = exp_args.env_args.max_steps or 0
        self._cumulative_tokens = 0
        self._cumulative_reward = 0.0
        self._is_tty = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()

        self._print(
            f"\n{'─' * 70}\n"
            f"  🚀  Starting: [{self._agent_name}] on [{self._task_name}]\n"
            f"{'─' * 70}",
            end="\n",
        )

    def on_step_start(self, step: int) -> None:
        self._step_start_time = time.monotonic()

    def on_action_selected(self, step: int, action: Optional[str], agent_info: dict) -> None:
        if self._show_action_preview and action:
            preview = action.replace("\n", " ")[: self._action_preview_chars]
            if len(action) > self._action_preview_chars:
                preview += "…"
            self._print(f"  ↳ action: {preview}", end="\n")

    def on_step_end(self, step_info: "StepInfo", is_done: bool) -> None:
        step = step_info.step
        reward = step_info.reward or 0.0
        self._cumulative_reward += reward

        # Accumulate tokens from step stats (populated by StepInfo.make_stats())
        step_tokens = 0
        if step_info.stats:
            step_tokens = step_info.stats.get("n_token_agent_messages", 0) or 0
        self._cumulative_tokens += step_tokens

        # Timing
        step_elapsed = time.monotonic() - self._step_start_time
        total_elapsed = time.monotonic() - self._exp_start_time

        # Step progress display
        max_steps_str = f"/{self._max_steps}" if self._max_steps else ""
        status_icon = "✓" if is_done and (step_info.terminated) else ("⏹" if is_done else "→")

        line = (
            f"  {status_icon}  step {step + 1:>3}{max_steps_str:<4}"
            f" | reward {self._cumulative_reward:>+6.2f}"
            f" | tokens {self._cumulative_tokens:>7,}"
            f" | step {step_elapsed:>5.1f}s"
            f" | elapsed {total_elapsed:>6.1f}s"
        )

        # Rewrite in-place when running in a real terminal; append newline in CI.
        end = "\n" if (is_done or not self._is_tty) else "\r"
        self._print(line, end=end)

    def on_exp_end(self, exp_dir, err_msg: Optional[str] = None) -> None:
        total_elapsed = time.monotonic() - self._exp_start_time
        status = "❌  ERROR" if err_msg else "✅  DONE"
        self._print(
            f"\n{'─' * 70}\n"
            f"  {status}  [{self._agent_name}] on [{self._task_name}]\n"
            f"  Total reward : {self._cumulative_reward:+.4f}\n"
            f"  Total tokens : {self._cumulative_tokens:,}\n"
            f"  Wall time    : {total_elapsed:.1f}s\n"
            f"  Output dir   : {exp_dir}\n"
            + (f"  Error        : {err_msg}\n" if err_msg else "")
            + f"{'─' * 70}\n",
            end="\n",
        )

    # ------------------------------------------------------------------
    # Internal print helper
    # ------------------------------------------------------------------

    def _print(self, msg: str, end: str = "\n") -> None:
        """Write to stderr so progress lines don't pollute captured stdout."""
        import sys

        print(msg, end=end, file=sys.stderr, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Built-in callback: BudgetLimitCallback
# ─────────────────────────────────────────────────────────────────────────────


class BudgetLimitCallback(AbstractCallback):
    """Safety guardrail that stops a runaway episode when cumulative token
    consumption exceeds a configured ceiling.

    When the budget is breached the callback raises :class:`BudgetExceededError`
    inside ``on_step_end``.  ``ExpArgs.run()``'s top-level ``except`` block
    catches this, records it in ``err_msg``, and proceeds to the ``finally``
    clean-up — so experiments, logs, and screenshots are still saved correctly.

    Args:
        max_tokens: Maximum number of *agent-message* tokens allowed across the
            full episode.  The count is sourced from
            ``step_info.stats["n_token_agent_messages"]`` (the same field that
            the existing ``StepInfo.make_stats()`` already populates).
        warn_at_pct: Emit a warning log when cumulative usage crosses this
            fraction of the budget (0–1).  Defaults to ``0.8`` (80 %).
    """

    def __init__(self, max_tokens: int, warn_at_pct: float = 0.8) -> None:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be a positive integer, got {max_tokens}.")
        if not (0.0 < warn_at_pct < 1.0):
            raise ValueError(f"warn_at_pct must be in (0, 1), got {warn_at_pct}.")

        self._max_tokens = max_tokens
        self._warn_threshold = int(max_tokens * warn_at_pct)
        self._cumulative_tokens: int = 0
        self._warned: bool = False

    def on_exp_start(self, exp_args: "ExpArgs") -> None:
        # Reset per episode so the same callback instance can be reused across
        # multiple sequential experiments in a batch.
        self._cumulative_tokens = 0
        self._warned = False

    def on_step_end(self, step_info: "StepInfo", is_done: bool) -> None:
        if is_done:
            return  # episode already ending; no need to enforce budget.

        step_tokens: int = 0
        if step_info.stats:
            step_tokens = step_info.stats.get("n_token_agent_messages", 0) or 0
        self._cumulative_tokens += step_tokens

        # Warn once when approaching the limit.
        if not self._warned and self._cumulative_tokens >= self._warn_threshold:
            logger.warning(
                "BudgetLimitCallback: token usage %s has reached %d%% of budget %s.",
                f"{self._cumulative_tokens:,}",
                int(self._warn_threshold / self._max_tokens * 100),
                f"{self._max_tokens:,}",
            )
            self._warned = True

        # Hard stop when budget is exceeded.
        if self._cumulative_tokens > self._max_tokens:
            raise BudgetExceededError(
                f"Token budget exceeded: {self._cumulative_tokens:,} > {self._max_tokens:,}. "
                "Episode aborted by BudgetLimitCallback."
            )


class BudgetExceededError(RuntimeError):
    """Raised by :class:`BudgetLimitCallback` when the token budget is breached.

    Caught by ``ExpArgs.run()``'s generic ``except Exception`` handler, which
    records it as ``err_msg`` and proceeds to clean up gracefully.
    """
