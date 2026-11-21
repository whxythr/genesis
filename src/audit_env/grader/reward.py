"""Terminal reward (official) - the only reward the environment ever returns.

    reward = clip(2 * F1 - 1 - STEP_COST * inspection_steps, -1, 1)

    inspection_steps = number of step() calls made BEFORE the submit step, including
                       calls that returned an error.  The submit itself counts towards the
                       100-action limit but not towards inspection_steps.  On truncation
                       (100 actions without submit, or the time limit) the findings are
                       empty and inspection_steps = steps taken.
    invalid findings structure -> reward = -1.

No rounding is applied: values are exact IEEE doubles of the formula (|error| < 1e-15).
"""

from __future__ import annotations

STEP_COST = 0.002
INVALID_REWARD = -1.0


def terminal_reward(f1: float, inspection_steps: int) -> float:
    value = 2.0 * f1 - 1.0 - STEP_COST * inspection_steps
    return max(-1.0, min(1.0, value))
