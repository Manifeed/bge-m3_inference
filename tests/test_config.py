from __future__ import annotations

from app.domain.config import DEFAULT_USE_FP16, resolve_use_fp16


def test_resolve_use_fp16_honors_explicit_true(monkeypatch) -> None:
    monkeypatch.setenv("BGE_M3_INFERENCE_USE_FP16", "true")

    assert resolve_use_fp16() is True


def test_resolve_use_fp16_honors_explicit_false(monkeypatch) -> None:
    monkeypatch.setenv("BGE_M3_INFERENCE_USE_FP16", "false")

    assert resolve_use_fp16() is False


def test_resolve_use_fp16_falls_back_to_default_for_invalid_value(monkeypatch) -> None:
    monkeypatch.setenv("BGE_M3_INFERENCE_USE_FP16", "unexpected")

    assert resolve_use_fp16() is DEFAULT_USE_FP16
