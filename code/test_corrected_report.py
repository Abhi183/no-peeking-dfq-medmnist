"""Tests for paired summaries used in the revised paper."""

from __future__ import annotations

import pandas as pd
import pytest

from corrected_report import (
    SYNTHETIC_DATA_SEED,
    SYNTHETIC_ROTATION_SEED,
    paired_deltas,
    summarize_metric,
)


def test_synthetic_data_and_rotation_use_independent_seeds():
    assert SYNTHETIC_DATA_SEED != SYNTHETIC_ROTATION_SEED


def test_paired_deltas_join_on_dataset_and_model_seed():
    frame = pd.DataFrame(
        [
            {"dataset": "a", "model_seed": 1, "method": "left", "auc": 0.7},
            {"dataset": "a", "model_seed": 2, "method": "left", "auc": 0.8},
            {"dataset": "a", "model_seed": 1, "method": "right", "auc": 0.9},
            {"dataset": "a", "model_seed": 2, "method": "right", "auc": 0.75},
        ]
    )
    deltas = paired_deltas(frame, "left", "right", metric="auc")
    assert deltas.tolist() == pytest.approx([0.2, -0.05])


def test_summary_uses_sample_standard_deviation():
    mean, std = summarize_metric(pd.Series([0.7, 0.8, 0.9]))
    assert mean == pytest.approx(0.8)
    assert round(std, 6) == 0.1
