"""Protocol tests for the corrected fixed-configuration evaluation."""

from __future__ import annotations

from corrected_eval import (
    ABLATION_METHODS,
    PRIMARY_METHODS,
    make_tensor_seed,
    parse_method,
)


def test_primary_protocol_is_fixed_and_data_free():
    assert PRIMARY_METHODS == (
        "uniform_full",
        "rotated_normal_codebook",
    )
    assert ABLATION_METHODS == (
        "uniform_full",
        "rotated_uniform",
        "normal_codebook",
        "rotated_normal_codebook",
    )


def test_method_parser_defines_two_by_two_cells():
    assert parse_method("uniform_full") == (False, "uniform")
    assert parse_method("rotated_uniform") == (True, "uniform")
    assert parse_method("normal_codebook") == (False, "normal_codebook")
    assert parse_method("rotated_normal_codebook") == (
        True,
        "normal_codebook",
    )


def test_tensor_seed_depends_only_on_rotation_seed_and_tensor_index():
    assert make_tensor_seed(rotation_seed=7, tensor_index=3) == make_tensor_seed(
        rotation_seed=7,
        tensor_index=3,
    )
    assert make_tensor_seed(rotation_seed=7, tensor_index=3) != make_tensor_seed(
        rotation_seed=8,
        tensor_index=3,
    )
    assert make_tensor_seed(rotation_seed=7, tensor_index=3) != make_tensor_seed(
        rotation_seed=7,
        tensor_index=4,
    )

