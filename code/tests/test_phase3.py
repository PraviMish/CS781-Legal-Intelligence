"""Offline tests for Phase-3 public code. No official data or weights required."""
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest
import torch
from torch import nn

from phase3.data import (OFFICIAL_LABELS, load_official, sentence_tokens,
                         context_tokens, collate, LengthBucketSampler, RRSentences)
from phase3.metrics import class_metrics
from phase3.context import transition_logprobs, viterbi
from phase3.submission import write_submission, read_submission, validate_predictions
from phase3.models import (LoRALinear, LoRMAPlusLinear, IA3Linear,
                           PrefixGQAAttention, LegalRRClassifier,
                           trainable_state, restore_trainable)
from phase1.models import GroupedQueryAttention, EfficientGQATransformer


def example_data(root):
    (root/'Train').mkdir()
    (root/'Test').mkdir()
    labels={name:i for i,name in enumerate(OFFICIAL_LABELS)}
    (root/'Train'/'label_vocab.json').write_text(json.dumps(labels))
    for domain in ('CL','IT'):
        for split in ('train','dev'):
            raw=[{'id':f'{domain}_{split}','text':['First.','Second.'],
                  'labels':['Fact','Issue']}]
            (root/'Train'/f'{domain}_{split}.json').write_text(json.dumps(raw))
        (root/'Test'/f'{domain}_test.json').write_text(
            json.dumps({f'{domain}_test':['One','Two','Three']}))
    return labels


def test_data_shapes_and_integrity(tmp_path):
    mapping=example_data(tmp_path)
    docs,labels=load_official(tmp_path)
    assert len(docs['CL_train'])==1
    assert labels[0]=='Fact'
    assert docs['IT_test'][0]['text']==['One','Two','Three']
    assert 'labels' not in docs['IT_test'][0]
    # strict label validation rejects malformed training annotations
    filename=tmp_path/'Train'/'CL_train.json'
    raw=json.loads(filename.read_text());raw[0]['labels']=[]
    filename.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_official(tmp_path)


def test_head_tail_truncation():
    seq=sentence_tokens(list(range(1,200)),128)
    assert len(seq)==128 and seq[0]==1 and seq[-1]==2
    assert seq[1]==1 and seq[-2]==199


def test_bidirectional_context_target_is_last():
    tokens=[[11,12],[21,22],[31,32]]
    seq=context_tokens(tokens,1,'bidirectional',128)
    assert seq==[1,11,12,2,31,32,2,21,22,2]
    assert context_tokens(tokens,0,'previous',128)==[1,11,12,2]


def test_batch_and_sampler():
    rows=[{'input_ids':[1,2,3][:i], 'label_id':i%13, 'domain':'CL',
           'doc_id':'x','sentence_idx':i} for i in (1,2,3)]
    batch=collate(rows)
    assert list(batch['lengths'])==[1,2,3]
    assert batch['input_ids'].shape==(3,3)
    sampler=LengthBucketSampler(RRSentences(rows),2,True,42)
    sampler.set_epoch(0)
    first=list(sampler)
    sampler.set_epoch(0)
    assert first==list(sampler)


def test_macro_includes_absent_classes():
    metrics=class_metrics([0,0,1],[0,1,1],13)
    assert abs(metrics['macro_f1']-(2/3+2/3)/13)<1e-10
    assert metrics['support'][2]==0


def test_viterbi_zero_weight_and_train_only_transitions():
    docs=[{'labels':['Fact','Fact','Issue']}]
    d={name:i for i,name in enumerate(OFFICIAL_LABELS)}
    s,t=transition_logprobs(docs,d)
    emissions=[[.3,.7]+[0]*11,[.8,.2]+[0]*11]
    assert viterbi(emissions,s,t,0)==[1,0]
    assert len(viterbi(emissions,s,t,.3))==2


def test_submission_zip(tmp_path):
    docs=[{'id':'x','text':['a','b']},{'id':'y','text':['c']}]
    preds={'x':['Fact','Issue'],'y':['Fact']}
    path=tmp_path/'submission.zip'
    write_submission(path,preds,docs,OFFICIAL_LABELS)
    assert read_submission(path)==preds
    with ZipFile(path) as z:
        assert z.namelist()==['prediction.json']
    with pytest.raises(ValueError):
        validate_predictions({'x':['Fact']},docs,OFFICIAL_LABELS)


@pytest.mark.parametrize('cls', [LoRALinear,LoRMAPlusLinear,IA3Linear])
def test_identity_initialized_adapters(cls):
    base=nn.Linear(16,32,bias=False)
    inputs=torch.randn(3,4,16)
    adapter=cls(base)
    with torch.no_grad():
        assert torch.allclose(adapter(inputs),base(inputs),atol=1e-6)
    adapter(inputs).sum().backward()
    assert any(p.grad is not None for p in adapter.parameters() if p.requires_grad)


def test_prefix_gqa_attention_shape():
    base=GroupedQueryAttention(64,4,2,64)
    prefix=PrefixGQAAttention(base,4)
    x=torch.randn(2,7,64)
    y,_=prefix(x)
    assert y.shape==x.shape
    y.sum().backward()
    assert prefix.prefix_value.grad is not None


def test_small_phase1_model_and_task_head():
    torch.manual_seed(42)
    foundation=EfficientGQATransformer(vocab_size=64,max_seq_len=64,d_model=64,
                                        n_layers=1,n_query_heads=4,n_kv_heads=2,d_ff=128)
    model=LegalRRClassifier(foundation,'efficient_gqa',64,13)
    inputs=torch.tensor([[1,7,2],[1,2,0]],dtype=torch.long)
    lengths=torch.tensor([3,2])
    logits=model(inputs,lengths)
    assert logits.shape==(2,13)
    assert torch.isfinite(logits).all()


def test_trainable_checkpoint_strict_names():
    model=nn.Linear(7,13)
    state=trainable_state(model)
    clone=nn.Linear(7,13)
    restore_trainable(clone,state)
    assert all(torch.equal(a,b) for a,b in zip(model.parameters(),clone.parameters()))
    with pytest.raises(ValueError):
        restore_trainable(clone,{'bogus':torch.randn(1)})


def test_best_only_architecture_matches_shared_phase1_module():
    """Small synthetic GQA foundation; not a claim about inaccessible original weights."""
    import importlib.util
    import sys
    package = Path(__file__).parents[2] / 'course_code' / 'phase3_best_model'
    # In a repository checkout the source-only winner is distributed separately;
    # this additional equivalence test is run when the best-only package is adjacent.
    if not package.is_dir():
        pytest.skip('Separate best-only package is not adjacent to repository')
    spec = importlib.util.spec_from_file_location('rr_phase3_winner', package / 'model.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    torch.manual_seed(11)
    kwargs = dict(vocab_size=64, max_seq_len=32, d_model=768,
                  n_layers=1, n_query_heads=12, n_kv_heads=4, d_ff=256)
    baseline = EfficientGQATransformer(**kwargs).eval()
    winner = runner.EfficientGQATransformer(**kwargs).eval()
    winner.load_state_dict(baseline.state_dict(), strict=True)
    tokens = torch.tensor([[1,7,13,2]], dtype=torch.long)
    with torch.inference_mode():
        h1,_ = baseline.hidden_states(tokens)
        h2,_ = winner.hidden_states(tokens)
    assert torch.allclose(h1,h2,atol=0,rtol=0)
