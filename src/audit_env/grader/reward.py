"""Terminal reward (official) — the only reward the environment ever returns.

    reward = clip(2 * F1 - 1 - STEP_COST * inspection_steps, -1, 1)
    inspection_steps = number of env steps taken before the submit step (including
                       steps that returned an error); on truncation = steps taken.
    invalid findings structure -> reward = -1.
"""

from __future__ import annotations

STEP_COST = 0.002
INVALID_REWARD = -1.0


def terminal_reward(f1: float, inspection_steps: int) -> float:
    value = 2.0 * f1 - 1.0 - STEP_COST * inspection_steps
    return round(max(-1.0, min(1.0, value)), 6)
