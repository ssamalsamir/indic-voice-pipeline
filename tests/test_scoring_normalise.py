"""Scoring normalisation must fold spelling variants WITHOUT folding real errors.

The risk with a scoring-time normaliser is that it quietly makes wrong transcripts
look right. These asserts pin both directions: variants collapse, genuine word
differences survive.

Run: .venv/bin/python -m tests.test_scoring_normalise
"""

from pipeline.text.normalise import normalise, normalise_for_scoring


def test_punctuation_does_not_cost_a_word():
    # The actual WER inflator: danda becomes "." and sticks to the token, so a
    # hypothesis without it scores the whole word wrong.
    assert normalise_for_scoring("यह सच है।") == normalise_for_scoring("यह सच है")


def test_nukta_variants_collapse():
    # क़िताब written with and without the nukta mark.
    assert normalise_for_scoring("क़िताब") == normalise_for_scoring("किताब")
    assert normalise_for_scoring("ज़रूर") == normalise_for_scoring("जरूर")


def test_anusvara_and_conjunct_nasal_collapse():
    assert normalise_for_scoring("संभव") == normalise_for_scoring("सम्भव")
    assert normalise_for_scoring("हिंदी") == normalise_for_scoring("हिन्दी")


def test_chandrabindu_folds_to_anusvara():
    assert normalise_for_scoring("हाँ") == normalise_for_scoring("हां")


def test_code_mixed_casing_is_not_an_error():
    assert normalise_for_scoring("मैंने Email भेजा") == normalise_for_scoring("मैंने email भेजा")


def test_real_word_errors_still_count():
    # The whole point: these must NOT collapse.
    assert normalise_for_scoring("यह सच है") != normalise_for_scoring("यह झूठ है")
    assert normalise_for_scoring("किताब") != normalise_for_scoring("किताबें")
    assert normalise_for_scoring("दस") != normalise_for_scoring("बीस")


def test_conjunct_fold_leaves_non_nasal_conjuncts_alone():
    # प्र is a real conjunct; folding it would corrupt the word.
    out = normalise_for_scoring("प्रकाश")
    assert "प्रकाश" == out, out


def test_corpus_normalise_keeps_punctuation():
    # The training corpus must NOT be stripped — only scoring is.
    assert "." in normalise("यह सच है।")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")
