"""Tests for the shared trace_id generator and error envelope (sciencerag/common)."""

import re

import pytest
from pydantic import ValidationError

from sciencerag.common.errors import ErrorResponse
from sciencerag.common.trace import new_trace_id

TRACE_ID_PATTERN = re.compile(r"^tr_\d{14}_[0-9a-f]{8}$")


def test_new_trace_id_matches_expected_format():
    assert TRACE_ID_PATTERN.match(new_trace_id())


def test_new_trace_id_is_unique_across_calls():
    ids = {new_trace_id() for _ in range(100)}
    assert len(ids) == 100


def test_new_trace_id_respects_custom_prefix():
    assert new_trace_id(prefix="run").startswith("run_")


def test_error_response_accepts_defined_category():
    resp = ErrorResponse(
        error={"category": "retrieval_timeout", "message": "corpus search timed out"},
        trace_id=new_trace_id(),
    )
    assert resp.status == "error"
    assert resp.error.category == "retrieval_timeout"


def test_error_response_rejects_undefined_category():
    with pytest.raises(ValidationError):
        ErrorResponse(
            error={"category": "not_a_real_category", "message": "x"},
            trace_id=new_trace_id(),
        )
