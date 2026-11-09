"""Atomic artifact/checkpoint test; no official private data required."""
import torch
import numpy as np
from common.artifacts import write_json, save_checkpoint, restore_checkpoint


def test_atomic_json(tmp_path):
    p=tmp_path/'temp.json'
    write_json(p,{'x':[1,2,3]})
    assert p.read_text().find('"x"') != -1
    assert not (tmp_path/'temp.json.tmp').exists()


def test_checkpoint_roundtrip(tmp_path):
    model=torch.nn.Linear(3,2)
    optim=torch.optim.AdamW(model.parameters(), lr=1e-3)
    rng=np.random.default_rng(42)
    checkpoint=tmp_path/'latest.pt'
    save_checkpoint(checkpoint, model, optim, step=5, tokens_seen=1234,
                    best_val_loss=2.0, sampler_rng=rng)
    recovered=torch.nn.Linear(3,2)
    state=restore_checkpoint(checkpoint, recovered, restore_rng=False)
    assert state['tokens_seen']==1234 and state['step']==5
    for a,b in zip(model.parameters(),recovered.parameters()):
        torch.testing.assert_close(a,b)
