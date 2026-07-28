"""Smoke + unit tests for the GPU-free core: config validation, text normalisation,
code-mixing, and metrics. These run in CI on any machine and guard the spine."""

from __future__ import annotations

import pytest

from pipeline.config import PipelineConfig
from pipeline.metrics import cer, corpus_wer, wer
from pipeline.text.codemix import code_mix_ratio, is_code_mixed
from pipeline.text.normalise import normalise


# -- config -------------------------------------------------------------------

def test_configs_load_and_validate():
    for name in ("hi_stt_kathbath", "hi_tts_indicvoices", "mr_stt_kathbath"):
        cfg = PipelineConfig.load(f"configs/{name}.yaml")
        assert cfg.run.name == name
        assert cfg.train.base_model  # never-from-scratch invariant holds


def test_tts_voice_clone_requires_consent(tmp_path):
    import yaml
    raw = yaml.safe_load(open("configs/hi_tts_indicvoices.yaml"))
    raw["data"].pop("consent")            # remove consent but keep voice_id
    with pytest.raises(ValueError, match="consent"):
        PipelineConfig.model_validate(raw)


# -- text normalisation -------------------------------------------------------

def test_devanagari_numerals_normalised():
    assert normalise("मैं १२३ हूँ", language="hi") == "मैं 123 हूँ"


def test_code_mixing_preserved_by_default():
    out = normalise("मैंने meeting cancel कर दी", keep_code_mixing=True)
    assert "meeting" in out and "cancel" in out


def test_code_mixing_can_be_stripped():
    out = normalise("मैंने meeting cancel कर दी", keep_code_mixing=False)
    assert "meeting" not in out


def test_nfc_and_whitespace_collapse():
    assert normalise("  a   b  ") == "a b"


# -- code-mix metrics ---------------------------------------------------------

def test_code_mix_ratio_and_flag():
    assert code_mix_ratio("मैं ठीक हूँ") == 0.0
    assert is_code_mixed("मैंने meeting cancel की")


# -- WER / CER ----------------------------------------------------------------

def test_wer_perfect_and_errors():
    assert wer("a b c", "a b c") == 0.0
    assert wer("a b c", "a b") == pytest.approx(1 / 3)


def test_cer_counts_characters():
    assert cer("कमल", "कमल") == 0.0
    assert cer("कमल", "कमज") == pytest.approx(1 / 3)


def test_corpus_wer_micro_average():
    pairs = [("a b", "a b"), ("c d", "c x")]
    assert corpus_wer(pairs) == pytest.approx(0.25)
