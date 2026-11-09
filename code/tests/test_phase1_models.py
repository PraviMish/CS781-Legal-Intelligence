"""Architecture size/forward/loss smoke tests against the original notebook."""
import gc
import torch
import pytest

from phase1.models import build_model, parameter_count

EXPECTED = {
    'decoder_only': 110615040,
    'encoder_decoder': 101972480,
    'prefix_lm': 110615040,
    'efficient_gqa': 112678656,
}


@pytest.mark.parametrize('architecture', list(EXPECTED))
def test_model_parameter_count_and_loss(architecture):
    model = build_model(architecture)
    assert parameter_count(model) == EXPECTED[architecture]
    with torch.no_grad():
        block = torch.randint(0, 32000, (1, 8), dtype=torch.long)
        loss = model.loss_on_block(block)
    assert loss.ndim == 0 and torch.isfinite(loss)
    del model
    gc.collect()


def test_efficient_gqa_cache_shapes():
    # Tiny model uses the same class / caching interface as the full model.
    from phase1.models import EfficientGQATransformer
    model = EfficientGQATransformer(vocab_size=48, max_seq_len=20,
                                    d_model=32, n_layers=2, n_query_heads=4,
                                    n_kv_heads=2, d_ff=64)
    model.eval()
    with torch.no_grad():
        logits, past = model(torch.tensor([[1, 2, 3]]), use_cache=True, last_only=True)
        assert logits.shape == (1, 1, 48)
        assert len(past) == 2
        _, continued = model(torch.tensor([[4]]), past_cache=past, use_cache=True, last_only=True)
    assert continued[0][0].shape[2] == 4
