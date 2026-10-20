"""
Tests for Zero Silent Defaults in DataPilot OS Resolver & Intent Schema.
Verifies that unrecognized operators and aggregations fail fast rather than guessing 'EQ' or 'MEAN'.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from backend.ds_engine.command_resolver import CommandResolver, command_resolver
from backend.ds_engine.intent_schema import FilterOperator, AggregationType


def test_unrecognized_operator_returns_none():
    """Verify resolve_operator returns None for invalid/unrecognized operators."""
    assert CommandResolver.resolve_operator("completely_unknown_operator_xyz") is None
    assert CommandResolver.resolve_operator("???") is None


def test_valid_operators_resolve_cleanly():
    """Verify valid operators resolve to correct FilterOperator enum."""
    assert CommandResolver.resolve_operator(">") == FilterOperator.GT
    assert CommandResolver.resolve_operator(">=") == FilterOperator.GTE
    assert CommandResolver.resolve_operator("<") == FilterOperator.LT
    assert CommandResolver.resolve_operator("<=") == FilterOperator.LTE
    assert CommandResolver.resolve_operator("==") == FilterOperator.EQ
    assert CommandResolver.resolve_operator("!=") == FilterOperator.NEQ
    assert CommandResolver.resolve_operator("contains") == FilterOperator.CONTAINS


def test_unrecognized_aggregation_returns_none():
    """Verify resolve_aggregation returns None for invalid aggregations."""
    assert CommandResolver.resolve_aggregation("unknown_agg_metric_123") is None
    assert CommandResolver.resolve_aggregation("invalid_aggregation_type") is None


def test_valid_aggregations_resolve_cleanly():
    """Verify valid aggregations resolve to canonical AggregationType enum."""
    assert CommandResolver.resolve_aggregation("sum") == AggregationType.SUM
    assert CommandResolver.resolve_aggregation("mean") == AggregationType.MEAN
    assert CommandResolver.resolve_aggregation("average") == AggregationType.MEAN
    assert CommandResolver.resolve_aggregation("count") == AggregationType.COUNT
    assert CommandResolver.resolve_aggregation("min") == AggregationType.MIN
    assert CommandResolver.resolve_aggregation("max") == AggregationType.MAX
    assert CommandResolver.resolve_aggregation("median") == AggregationType.MEDIAN


def test_filter_dict_with_invalid_operator_rejected():
    """Verify normalize_filter_spec drops/rejects filters with invalid operators."""
    invalid_filter = {"column": "sales", "operator": "invalid_op_xyz", "value": 100}
    spec = CommandResolver.normalize_filter_spec(invalid_filter)
    assert spec is None
