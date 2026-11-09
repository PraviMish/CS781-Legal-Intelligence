"""Decoder's schema/word-count safeguards, optional tokenizer dependency."""
import pytest
pytest.importorskip('tokenizers')
from phase1.generate import (
    GREEDY_FULL_PROMPT, LEGACY, SWEEP_LEGACY, LegalDecoder, banned_ngram_tokens, exactly_128_words,
)
from phase1.submit import validate_predictions


def test_word_count():
    assert exactly_128_words(' '.join(['word']*127)) is None
    assert len(exactly_128_words(' '.join(['word']*200)).split())==128


def test_ngram_guard():
    assert banned_ngram_tokens([1,2,3,4,1,2,3],4)=={4}


def test_config_preserves_original_winner():
    assert LEGACY.strategy=='sample'
    assert LEGACY.min_generation_tokens == 560
    assert GREEDY_FULL_PROMPT.strategy=='greedy'


@pytest.mark.parametrize(('config', 'retry_name', 'seed_offset'), [
    (LEGACY, LEGACY.name, 7919),
    (GREEDY_FULL_PROMPT, 'greedy_full_prompt_emergency', 0),
    (SWEEP_LEGACY, 'legacy_guarded_emergency', 0),
])
def test_notebook_retry_policy(monkeypatch, config, retry_name, seed_offset):
    decoder = LegalDecoder.__new__(LegalDecoder)
    attempts = []

    def generate_once(model, architecture, prompt, attempt):
        attempts.append(attempt)
        if len(attempts) == 1:
            raise RuntimeError('Fewer than 128 words')
        return 'completed'

    monkeypatch.setattr(decoder, '_generate_once', generate_once)
    assert decoder.generate(None, 'efficient_gqa', 'prompt', config) == 'completed'
    retry = attempts[1]
    assert retry.name == retry_name
    assert retry.seed_offset == seed_offset
    assert retry.strategy == 'sample'
    assert retry.min_generation_tokens == 560
    assert retry.max_tokens_without_word == 1


def test_validation_of_submission_structure():
    official={'id_1':'prompt','id_2':'prompt'}
    good={key:{'text':'word '*128} for key in official}
    validate_predictions(good,official)
    with pytest.raises(ValueError):
        validate_predictions({'id_1':good['id_1']},official)
