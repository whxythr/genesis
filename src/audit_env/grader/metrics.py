"""Precision / recall / F1 with the published edge cases."""

from __future__ import annotations


def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if tp + fp + fn == 0:
        f1 = 1.0  # empty truth and empty findings
    else:
        f1 = 2 * tp / (2 * tp + fp + fn)  # empty truth + any finding -> 0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)}
