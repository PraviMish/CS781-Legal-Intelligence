"""Prior Case Retrieval E00-E06: original Phase 1 initialized neural experiments

Faithful refactoring of executed CS781 Capstone Phase 7 notebook code.
Original experimental algorithms and hyperparameters are preserved.
CLI replaces Colab Drive mounting and notebook class restoration.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from tempfile import gettempdir

from phase7.validation import split_directory

def main(argv=None):
    parser=argparse.ArgumentParser(description='Phase 7 original model adaptations E00--E06')
    parser.add_argument('--project-root',type=Path,default=Path.cwd())
    parser.add_argument('--data-dir',type=Path)
    parser.add_argument('--checkpoint',type=Path,help='Phase 1 Efficient GQA .pt containing model_state')
    parser.add_argument('--tokenizer',type=Path,help='Phase 1 byte-level BPE JSON')
    parser.add_argument('--output-dir',type=Path,help='Directory for all original Phase 7 artifacts')
    args=parser.parse_args(argv)
    # -- Notebook source block 3 --
    import os, sys, json, math, time, random, gc, re, zipfile, hashlib, pickle, contextlib
    from collections import Counter, defaultdict
    import numpy as np, pandas as pd
    from IPython.display import display
    from tqdm.auto import tqdm
    import torch, torch.nn as nn, torch.nn.functional as F
    from tokenizers import Tokenizer
    from huggingface_hub import HfApi, hf_hub_download
    from sklearn.feature_extraction.text import CountVectorizer
    from scipy.sparse import coo_matrix, csr_matrix

    VERSION='p7_final_v1'
    SEED=230787
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(SEED)
    ROOT=args.project_root.expanduser().resolve()
    RUNTIME=Path(os.environ.get('PHASE7_RUNTIME', Path(gettempdir()) / 'phase7_runtime'))
    HF_TOKEN=os.environ.get('HF_TOKEN')
    RUNTIME.mkdir(parents=True,exist_ok=True)
    P1=ROOT/'phase1'; P7=ROOT/'phase7'
    DIR={name:P7/name for name in ('data','cache','models','logs','submissions')}
    if args.data_dir:DIR['data']=args.data_dir.expanduser().resolve()
    if args.output_dir:P7=args.output_dir.expanduser().resolve();DIR={name:P7/name for name in ('data','cache','models','logs','submissions')}
    if args.data_dir:DIR['data']=args.data_dir.expanduser().resolve()
    for d in DIR.values(): d.mkdir(parents=True,exist_ok=True)
    DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    AMP_DTYPE=torch.bfloat16 if DEVICE.type=='cuda' and torch.cuda.is_bf16_supported() else torch.float16
    if DEVICE.type=='cuda':torch.backends.cuda.matmul.allow_tf32=True

    CFG={
        'training_epochs':2,  # ALL labeled training queries, every epoch; NO QUERY CAP
        'batch_queries':1,    # A100-memory-safe; gradient accumulated below
        'grad_accum':8,
        'save_every_updates':80,
        'chunk_tokens':192,
        'chunks_per_document':2,
        'embedding_dim':256,
        'shortlist_k':256,
        'eval_batch_docs':8,
        'temperature':0.07,
        'weight_decay':0.01,
        'lorma_rank':16,
        'lorma_alpha':32,
        'student_layers':6,
        'kd_temperature':2.0,
        'kd_weight':0.40,
        'evaluation_weights':[0.0,0.15,0.3,0.45,0.6,0.75,0.9,1.0],
        'rrf_k':60,
    }

    def atomic_json(obj,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_name(path.name+'.tmp')
        with open(tmp,'w',encoding='utf-8') as f:json.dump(obj,f,indent=2,ensure_ascii=False)
        os.replace(tmp,path)

    def atomic_torch(obj,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_name(path.name+'.tmp')
        torch.save(obj,tmp);os.replace(tmp,path)

    def amp():
        return torch.autocast('cuda',dtype=AMP_DTYPE) if DEVICE.type=='cuda' else contextlib.nullcontext()

    print('Version:',VERSION,'Device:',DEVICE)
    if DEVICE.type=='cuda': print('GPU:',torch.cuda.get_device_name(0),'Memory GB:',round(torch.cuda.get_device_properties(0).total_memory/2**30,2))
    print('Project:',ROOT,'\nPhase 7:',P7)
    print('Training cap: NONE. Full labeled query set each epoch.')

    # -- Notebook source block 5 --
    REPO='Exploration-Lab/CS-781-Capstone'
    SPLITS=('train','dev','test')
    FILES={s:(f'{s}.json',f'{s}_candidates.json') for s in SPLITS}
    PATHS={(s,name):split_directory(DIR['data'], s) / name for s in SPLITS for name in FILES[s]}
    for p in PATHS.values():p.parent.mkdir(parents=True,exist_ok=True)
    missing=[(s,n) for (s,n),p in PATHS.items() if not p.exists() or p.stat().st_size==0]
    if missing:
        api=HfApi()
        repo_files=api.list_repo_files(repo_id=REPO,repo_type='dataset',token=HF_TOKEN)
        available=[p for p in repo_files if 'phase_7_pcr/' in p.lower() and p.lower().endswith('.json')]
        print('Phase 7 repository files:',*available,sep='\n  ')
        for split,name in missing:
            matches=[p for p in available if p.lower().endswith('/'+split+'/'+name.lower())]
            assert len(matches)==1,(split,name,matches)
            download=hf_hub_download(repo_id=REPO,filename=matches[0],repo_type='dataset',token=HF_TOKEN)
            dest=PATHS[split,name]
            tmp=dest.with_name(dest.name+'.tmp')
            import shutil
            shutil.copy2(download,tmp);os.replace(tmp,dest)
            print('Saved:',dest,round(dest.stat().st_size/2**20,2),'MB')
    else: print('Reusing all six official JSON files already on Drive.')
    assert all(p.exists() and p.stat().st_size for p in PATHS.values())
    DATA_SIGNATURE=hashlib.sha256(json.dumps([(s,n,PATHS[s,n].stat().st_size) for s in SPLITS for n in FILES[s]],sort_keys=True).encode()).hexdigest()[:16]
    print('Data signature:',DATA_SIGNATURE)

    # -- Notebook source block 6 --
    def read_json(path):
        with open(path,encoding='utf-8-sig') as f:return json.load(f)
    RAW={s:{name:read_json(PATHS[s,name]) for name in FILES[s]} for s in SPLITS}
    for s in SPLITS:
        for name in FILES[s]:
            obj=RAW[s][name]
            print('\n',s,name,'type=',type(obj).__name__,'count=',len(obj))
            first=(next(iter(obj.items())) if isinstance(obj,dict) else obj[0])
            print(json.dumps(first,ensure_ascii=False,default=str)[:1250])

    # -- Notebook source block 8 --
    ID_KEYS=('id','case_id','query_id','document_id','doc_id')
    TEXT_KEYS=('text','sentences','document','content','case_text')
    GOLD_KEYS=('relevant_candidates','relevant_case_ids','relevant_ids','gold')

    def as_text(x):
        if x is None:return ''
        if isinstance(x,str):return x.strip()
        if isinstance(x,(int,float)):return str(x)
        if isinstance(x,list):return '\n'.join(filter(None,(as_text(a) for a in x)))
        if isinstance(x,dict):
            for key in ('sentence','text','content','document'):
                if key in x:return as_text(x[key])
            return ' '.join(filter(None,(as_text(a) for a in x.values())))
        raise TypeError(type(x))

    def items(raw):
        if isinstance(raw,list):
            for i,r in enumerate(raw):yield str(i),r
        elif isinstance(raw,dict):
            if len(raw)==1:
                key=next(iter(raw))
                if key in ('data','records','cases','queries','candidates','items'):
                    yield from items(raw[key]);return
            for k,v in raw.items():yield str(k),v
        else:raise TypeError('Expected list/dict JSON, got '+str(type(raw)))

    def get_id(r,external):
        if isinstance(r,dict):
            for k in ID_KEYS:
                if k in r and r[k] is not None:return str(r[k])
        return str(external)

    def get_text(r):
        if not isinstance(r,dict):return as_text(r)
        for k in TEXT_KEYS:
            if k in r:return as_text(r[k])
        raise ValueError('No text field among '+str(list(r)))

    def get_gold(r):
        if not isinstance(r,dict):return []
        value=next((r[k] for k in GOLD_KEYS if k in r),None)
        if value is None:return []
        if isinstance(value,dict):
            return [str(k) for k,v in value.items() if v]
        if isinstance(value,(str,int)):return [str(value)]
        if not isinstance(value,list):raise ValueError('Unexpected gold type: '+str(type(value)))
        vals=[]
        for g in value:
            if isinstance(g,dict):
                found=next((str(g[k]) for k in ID_KEYS if k in g),None)
                if found is None:raise ValueError('Gold item has no ID: '+str(g)[:200])
                vals.append(found)
            else:vals.append(str(g))
        return list(dict.fromkeys(vals))

    def normalize(raw,is_query):
        out={}
        for k,r in items(raw):
            rid=get_id(r,k)
            if rid in out:raise ValueError('Duplicate case ID: '+rid)
            row={'id':rid,'text':get_text(r)}
            if is_query:row['gold']=get_gold(r)
            out[rid]=row
        return out

    DATA={}
    for s in SPLITS:
        Q=normalize(RAW[s][f'{s}.json'],True)
        C=normalize(RAW[s][f'{s}_candidates.json'],False)
        assert Q and len(C)>=7,(s,len(Q),len(C))
        assert all(row['text'].strip() for row in Q.values()),(s,'empty query')
        assert all(row['text'].strip() for row in C.values()),(s,'empty candidate')
        # Resolve only unambiguous leading-zero mismatches; preserve original candidate ID.
        lookup=defaultdict(list)
        for cid in C:lookup[cid.lstrip('0') or '0'].append(cid)
        missing=[]
        for q in Q.values():
            resolved=[]
            for gold in q['gold']:
                if gold in C:resolved.append(gold);continue
                matches=lookup[gold.lstrip('0') or '0']
                if len(matches)==1:resolved.append(matches[0])
                else:missing.append((q['id'],gold))
            q['gold']=list(dict.fromkeys(resolved))
        if missing:raise ValueError(f'{s}: gold candidate IDs absent/ambiguous: {missing[:10]}')
        DATA[s]={'queries':Q,'candidates':C}
    assert all(not q['gold'] for q in DATA['test']['queries'].values()),'Public test has labels; never train on test.'
    rows=[]
    for s,collection in DATA.items():
        qs=list(collection['queries'].values());cs=list(collection['candidates'].values())
        rows.append({'split':s,'queries':len(qs),'candidates':len(cs),
                     'query_median_words':int(np.median([len(q['text'].split()) for q in qs])),
                     'candidate_p95_words':int(np.percentile([len(c['text'].split()) for c in cs],95)),
                     'labelled_queries':sum(bool(q['gold']) for q in qs) if s!='test' else None,
                     'gold_pairs':sum(len(q['gold']) for q in qs) if s!='test' else None})
    print('DATA INTEGRITY PASSED');display(pd.DataFrame(rows));atomic_json(rows,DIR['logs']/'dataset_summary.json')

    # -- Notebook source block 10 --
    def eval_top7(pred,split='dev'):
        Q=DATA[split]['queries'];C=DATA[split]['candidates']
        assert set(pred)==set(Q),(split,'missing/extra query IDs')
        hits=0;gold_n=0
        for qid,ranked in pred.items():
            assert isinstance(ranked,list) and len(ranked)==7 and len(set(ranked))==7,qid
            assert all(isinstance(cid,str) and cid in C for cid in ranked),qid
            gold=set(Q[qid]['gold']);hits+=len(gold.intersection(ranked));gold_n+=len(gold)
        p=hits/(7*len(Q));r=hits/max(1,gold_n)
        return {'micro_f1_at7':2*hits/max(1,7*len(Q)+gold_n),
                'precision_at7':p,'recall_at7':r,'hits':hits,'gold_pairs':gold_n,'n_queries':len(Q)}

    def submit(pred,name):
        Q=DATA['test']['queries'];C=DATA['test']['candidates']
        assert set(pred)==set(Q)
        for q,ranked in pred.items():
            assert isinstance(q,str) and isinstance(ranked,list) and len(ranked)==7,q
            assert len(set(ranked))==7 and all(isinstance(x,str) and x in C for x in ranked),q
        output=DIR['submissions']/name;output.mkdir(parents=True,exist_ok=True)
        jsonpath=output/'prediction.json';zippath=output/'submission.zip'
        atomic_json(pred,jsonpath)
        tmp=zippath.with_name(zippath.name+'.tmp')
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:z.write(jsonpath,'prediction.json')
        os.replace(tmp,zippath)
        with zipfile.ZipFile(zippath) as z:
            assert z.testzip() is None and z.namelist()==['prediction.json']
            assert json.loads(z.read('prediction.json'))==pred
        print('VALID ZIP:',name,'queries=',len(Q),'path=',zippath)
        return zippath
    print('Submission validation functions ready.')

    # -- Notebook source block 12 --
    class BM25Index:
        def __init__(self,mode='phrase'):
            self.mode=mode
            self.vectorizer=CountVectorizer(lowercase=True,token_pattern=r'(?u)\b\w+\b',
                      ngram_range=(1,2) if mode=='phrase' else (1,1),
                      max_features=130_000,dtype=np.float32)
        def fit(self,texts):
            tf=self.vectorizer.fit_transform(texts).tocsr()
            lens=np.asarray(tf.sum(axis=1)).ravel().astype(np.float32)
            avg=max(1.0,float(np.mean(lens)))
            df=np.asarray((tf>0).sum(axis=0)).ravel()
            n=tf.shape[0];idf=np.log1p((n-df+.5)/(df+.5)).astype(np.float32)
            coo=tf.tocoo()
            normal=1.2*(.25+.75*lens[coo.row]/avg)
            weights=(coo.data*2.2/(coo.data+normal+1e-8))*idf[coo.col]
            self.mat=csr_matrix((weights,(coo.row,coo.col)),shape=tf.shape,dtype=np.float32)
            return self
        def score(self,text):
            q=self.vectorizer.transform([text]).tocsr()
            q.data[:]=1.0
            return np.asarray((self.mat@q.T).toarray()).ravel()

    BM25={}
    for split in SPLITS:
        cache=DIR['cache']/f'{VERSION}_{DATA_SIGNATURE}_bm25_{split}_{CFG["shortlist_k"]}.pkl'
        if cache.exists():
            BM25[split]=pickle.load(open(cache,'rb'))
            if set(BM25[split])==set(DATA[split]['queries']):
                print('Restored BM25:',split);continue
        candidates=DATA[split]['candidates'];ids=list(candidates)
        print('Indexing',split,len(ids),'candidate documents (word + phrase BM25)')
        index=BM25Index('phrase').fit([candidates[x]['text'] for x in ids])
        ans={};K=min(len(ids),max(7,CFG['shortlist_k']))
        for qid,q in tqdm(DATA[split]['queries'].items(),desc='BM25 '+split):
            scores=index.score(q['text'])
            order=np.argpartition(-scores,K-1)[:K]
            order=order[np.argsort(-scores[order],kind='stable')]
            ans[qid]=[ids[int(i)] for i in order]
        BM25[split]=ans
        with open(cache,'wb') as f:pickle.dump(ans,f,protocol=pickle.HIGHEST_PROTOCOL)
        del index;gc.collect()

    bm25_dev={qid:v[:7] for qid,v in BM25['dev'].items()}
    BM25_METRICS=eval_top7(bm25_dev,'dev')
    print('BM25 dev metrics:');display(pd.DataFrame([BM25_METRICS]))

    def oracle(split='dev'):
        q=DATA[split]['queries'];hits=0;gold=0;available=0
        for qid,row in q.items():
            g=set(row['gold']);intersection=len(g.intersection(BM25[split][qid]))
            available+=intersection;hits+=min(7,intersection);gold+=len(g)
        return {'shortlist_recall':available/max(gold,1),'oracle_micro_f1_at7':2*hits/max(7*len(q)+gold,1)}
    ORACLE=oracle();print('Shortlist diagnostics:',ORACLE)
    submit({qid:v[:7] for qid,v in BM25['test'].items()},'E00_BM25')
    atomic_json({'baseline':BM25_METRICS,'oracle':ORACLE},DIR['logs']/'bm25_metrics.json')

    # -- Notebook source block 14 --
    from phase1.models import EfficientGQATransformer
    TOKENIZER_PATH=args.tokenizer.expanduser().resolve() if args.tokenizer else P1/'tokenizer/legal_bpe_32k.json'
    BASE_PATH=args.checkpoint.expanduser().resolve() if args.checkpoint else P1/'models/efficient_gqa/milestones/best_up_to_700M.pt'
    assert TOKENIZER_PATH.is_file(),TOKENIZER_PATH
    assert BASE_PATH.is_file(),BASE_PATH
    TOKENIZER=Tokenizer.from_file(str(TOKENIZER_PATH))
    VOCAB=TOKENIZER.get_vocab_size();PAD=TOKENIZER.token_to_id('<pad>');UNK=TOKENIZER.token_to_id('<unk>')
    assert VOCAB==32000 and PAD is not None and UNK is not None
    print('Loading original Phase 1 checkpoint (CPU)...')
    base=torch.load(BASE_PATH,map_location='cpu',weights_only=False)
    assert 'model_state' in base,base.keys()
    BASE_STATE=base['model_state']
    assert isinstance(BASE_STATE,dict) and 'token_embedding.weight' in BASE_STATE
    print('Checkpoint keys:',list(base.keys()),'vocab:',VOCAB,'weight tensor count:',len(BASE_STATE))
    del base;gc.collect()
    def new_backbone(layers=14):
        model=EfficientGQATransformer(vocab_size=VOCAB,max_seq_len=1280,d_model=768,
                              n_layers=layers,n_query_heads=12,n_kv_heads=4,d_ff=2048)
        if layers==14:
            model.load_state_dict(BASE_STATE,strict=True)
        else:
            source_layers=np.linspace(0,13,layers).round().astype(int).tolist()
            state={}
            for name in model.state_dict():
                parts=name.split('.')
                src=name
                if len(parts)>2 and parts[0]=='blocks':
                    parts[1]=str(source_layers[int(parts[1])]);src='.'.join(parts)
                assert src in BASE_STATE,(name,src)
                state[name]=BASE_STATE[src]
            model.load_state_dict(state,strict=True)
            print('Student initialized from Phase 1 block indices:',source_layers)
        return model
    probe=new_backbone(14)
    assert len(probe.blocks)==14 and hasattr(probe,'hidden_states')
    print('Phase 1 weights strictly loaded, params:',sum(p.numel() for p in probe.parameters()))
    del probe;gc.collect()

    # -- Notebook source block 16 --
    from functools import lru_cache
    L=CFG['chunk_tokens'];NC=CFG['chunks_per_document']
    @lru_cache(maxsize=100000)
    def chunks(split,role,rid):
        text=DATA[split][role][rid]['text']
        ids=TOKENIZER.encode(text,add_special_tokens=False).ids or [UNK]
        starts=sorted(set(np.linspace(0,max(0,len(ids)-L),NC).astype(int).tolist()))
        out=[tuple(ids[i:i+L]) for i in starts]
        return tuple((out+[tuple()]*(NC-len(out)))[:NC])

    def batch_tokens(refs,device=DEVICE):
        x=torch.full((len(refs),NC,L),PAD,dtype=torch.long)
        for j,(split,role,rid) in enumerate(refs):
            for ci,chunk in enumerate(chunks(split,role,rid)):
                if chunk:x[j,ci,:len(chunk)]=torch.tensor(chunk,dtype=torch.long)
        return x.to(device)

    class RetrievalModel(nn.Module):
        def __init__(self,backbone):
            super().__init__();self.backbone=backbone
            self.projection=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,CFG['embedding_dim']),nn.Tanh())
        def forward(self,token_ids):
            B,C,T=token_ids.shape
            x=token_ids.reshape(B*C,T)
            hidden,_=self.backbone.hidden_states(x)
            valid=x.ne(PAD)
            chunk_vec=(hidden*valid[...,None]).sum(1)/valid.sum(1,keepdim=True).clamp(min=1)
            chunk_vec=chunk_vec.reshape(B,C,-1)
            cv=valid.any(1).reshape(B,C)
            doc=(chunk_vec*cv[...,None]).sum(1)/cv.sum(1,keepdim=True).clamp(min=1)
            return F.normalize(self.projection(doc.float()),p=2,dim=-1)

    TRAIN_ALL=list(DATA['train']['queries'])
    TRAIN_LABELLED=[q for q in TRAIN_ALL if DATA['train']['queries'][q]['gold']]
    TRAIN_UNLABELLED=[q for q in TRAIN_ALL if not DATA['train']['queries'][q]['gold']]
    assert TRAIN_LABELLED,'No labeled training queries in data!'
    print('FULL TRAINING SET:',len(TRAIN_ALL),'queries',
          'labeled:',len(TRAIN_LABELLED),'unlabeled:',len(TRAIN_UNLABELLED))
    print('No max_train_queries restriction is used.')
    TRAIN_CIDS=list(DATA['train']['candidates'])

    def training_triplet(qid,epoch,position):
        rng=random.Random(SEED+epoch*10000019+position*1009)
        gold=set(DATA['train']['queries'][qid]['gold'])
        positives=sorted(gold)
        pos=positives[(epoch+position)%len(positives)]
        hard=[c for c in BM25['train'][qid][:75] if c not in gold and c!=qid]
        if not hard:
            hard=[c for c in TRAIN_CIDS if c not in gold and c!=qid]
        h=rng.choice(hard[:min(25,len(hard))])
        excluded=gold|{qid,h}
        negative=None
        for _ in range(100):
            c=rng.choice(TRAIN_CIDS)
            if c not in excluded:negative=c;break
        if negative is None:raise RuntimeError('Unable to sample random negative for '+qid)
        return [pos,h,negative]

    example=TRAIN_LABELLED[0]
    ex=training_triplet(example,0,0)
    assert ex[0] in DATA['train']['queries'][example]['gold']
    assert ex[1] not in DATA['train']['queries'][example]['gold'] and ex[2]!=ex[1]
    print('Triplet:',example,ex)
    print('Token tensor:',batch_tokens([('train','queries',example)],device='cpu').shape)

    # -- Notebook source block 18 --
    class IA3Linear(nn.Module):
        def __init__(self,base,location):
            super().__init__();self.base=base;self.location=location
            for p in base.parameters():p.requires_grad=False
            n=base.in_features if location=='input' else base.out_features
            self.scale=nn.Parameter(torch.ones(n))
        def forward(self,x):
            return self.base(x*self.scale) if self.location=='input' else self.base(x)*self.scale

    class LoRMAPlus(nn.Module):
        def __init__(self,base,rank=16,alpha=32):
            super().__init__();self.base=base;self.factor=alpha/rank
            for p in base.parameters():p.requires_grad=False
            self.A=nn.Linear(base.out_features,rank,bias=False)
            self.B=nn.Linear(rank,base.out_features,bias=False)
            nn.init.kaiming_uniform_(self.A.weight,a=math.sqrt(5));nn.init.zeros_(self.B.weight)
        def forward(self,x):
            z=self.base(x)
            return z+self.factor*self.B(self.A(z))

    EXPS={
      'E01_PARTIAL':{'method':'partial','lr':3e-5},
      'E02_IA3':{'method':'ia3','lr':4e-4},
      'E03_LORMA':{'method':'lorma','lr':2e-4},
      'E04_FULL':{'method':'full','lr':1.5e-5},
      'E05_KD':{'method':'kd','lr':2e-5},
    }

    def build_experiment(exp):
        method=EXPS[exp]['method']
        backbone=new_backbone(CFG['student_layers'] if method=='kd' else 14)
        if method in ('partial','ia3','lorma'):
            for p in backbone.parameters():p.requires_grad=False
        if method=='partial':
            for block in backbone.blocks[-4:]:
                for p in block.parameters():p.requires_grad=True
            for p in backbone.final_norm.parameters():p.requires_grad=True
        elif method=='ia3':
            for b in backbone.blocks:
                for name in ('k_proj','v_proj'):
                    setattr(b.attention,name,IA3Linear(getattr(b.attention,name),'output'))
                b.mlp.down=IA3Linear(b.mlp.down,'input')
        elif method=='lorma':
            for b in backbone.blocks:
                for name in ('q_proj','v_proj','out_proj'):
                    setattr(b.attention,name,LoRMAPlus(getattr(b.attention,name),CFG['lorma_rank'],CFG['lorma_alpha']))
                b.mlp.down=LoRMAPlus(b.mlp.down,CFG['lorma_rank'],CFG['lorma_alpha'])
        elif method not in ('full','kd'):raise ValueError(method)
        model=RetrievalModel(backbone)
        return model

    def params(model):
        return {'parameters_m':round(sum(p.numel() for p in model.parameters())/1e6,3),
                'trainable_m':round(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6,3)}

    # Verify two adaptation variants are precisely identity maps at initialization.
    test=nn.Linear(9,7,bias=False);x=torch.randn(2,9)
    assert torch.allclose(LoRMAPlus(test)(x),test(x),atol=1e-6)
    assert torch.allclose(IA3Linear(test,'output')(x),test(x),atol=1e-6)
    print('Adapter identity checks PASSED.')

    # -- Notebook source block 20 --
    assert DEVICE.type=='cuda','Please select an A100 GPU in Colab Runtime before continuing.'
    assert torch.cuda.is_bf16_supported(),'Use A100, H100 or another BF16 GPU for this notebook.'
    for exp in EXPS:
        print('\nPreflight:',exp)
        model=build_experiment(exp).to(DEVICE)
        model.train()
        refs=[('train','queries',TRAIN_LABELLED[0])]
        cids=training_triplet(TRAIN_LABELLED[0],0,0)
        cx=[('train','candidates',c) for c in cids]
        with amp():
            q=model(batch_tokens(refs));c=model(batch_tokens(cx))
            logit=(q@c.T).float()/CFG['temperature']
            loss=F.cross_entropy(logit,torch.tensor([0],device=DEVICE))
        assert logit.shape==(1,3) and torch.isfinite(loss)
        loss.backward()
        changed=[p for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert changed and all(torch.isfinite(p.grad).all().item() for p in changed)
        print('  logits',list(logit.shape),'loss',round(float(loss),4),params(model),'gradient tensors:',len(changed))
        del model,q,c,logit,loss;gc.collect();torch.cuda.empty_cache()
    print('\nALL FIVE ARCHITECTURE/PARAMETER PRE-FLIGHTS PASSED.')

    # -- Notebook source block 22 --
    @torch.no_grad()
    def embedding_batches(model,refs,desc='Embedding'):
        model.eval();out=[];b=CFG['eval_batch_docs']
        for start in tqdm(range(0,len(refs),b),desc=desc,leave=False):
            ids=batch_tokens(refs[start:start+b]);
            with amp():vec=model(ids)
            out.append(vec.float().cpu().numpy())
        return np.concatenate(out,axis=0).astype(np.float32)

    def neural_ranks(model,split,exp=None,cache_test=False):
        qids=list(DATA[split]['queries']);cids=sorted({cid for r in BM25[split].values() for cid in r})
        def embeddings(role,ids):
            cache=None
            if cache_test and exp:
                best=DIR['models']/exp/'best.pt';st=best.stat()
                key=hashlib.sha256(f'{VERSION}:{DATA_SIGNATURE}:{exp}:{best.stat().st_size}:{st.st_mtime_ns}:{split}:{role}:{L}:{NC}:{len(ids)}'.encode()).hexdigest()[:16]
                folder=DIR['cache']/'embeddings'/exp;folder.mkdir(parents=True,exist_ok=True)
                cache=folder/f'{split}_{role}_{key}.npy'
                if cache.exists():
                    v=np.load(cache,mmap_mode='r')
                    if v.shape==(len(ids),CFG['embedding_dim']):return np.asarray(v)
            vec=embedding_batches(model,[(split,role,i) for i in ids],f'{exp or "model"} {split} {role}')
            if cache is not None:
                tmp=str(cache)+'.tmp'
                with open(tmp,'wb') as f:np.save(f,vec)
                os.replace(tmp,cache)
            return vec
        Q=embeddings('queries',qids);C=embeddings('candidates',cids)
        pos={cid:i for i,cid in enumerate(cids)}
        ranks={}
        for i,qid in enumerate(qids):
            shortlist=BM25[split][qid];ix=[pos[c] for c in shortlist]
            scores=C[ix]@Q[i]
            sorted_idx=np.argsort(-scores,kind='stable')
            dense_rank=np.empty(len(ix),dtype=np.int32);dense_rank[sorted_idx]=np.arange(1,len(ix)+1)
            ranks[qid]=dense_rank
        return ranks

    def ranked_predictions(split,dense_ranks,lex_weight):
        ans={};k=CFG['rrf_k']
        for qid,shortlist in BM25[split].items():
            dr=dense_ranks[qid]
            br=np.arange(1,len(shortlist)+1)
            scores=lex_weight/(k+br)+(1-lex_weight)/(k+dr)
            order=np.argsort(-scores,kind='stable')[:7]
            ans[qid]=[shortlist[int(i)] for i in order]
        return ans

    def validate_dev(model,exp):
        ranks=neural_ranks(model,'dev',exp)
        results=[]
        for w in CFG['evaluation_weights']:
            metrics=eval_top7(ranked_predictions('dev',ranks,w))
            results.append({'bm25_weight':w,**metrics})
        df=pd.DataFrame(results).sort_values('micro_f1_at7',ascending=False,kind='stable')
        best=df.iloc[0].to_dict()
        print('Dev',exp,'best:',best)
        display(df)
        # Keep all dense ranks for later optional ensemble, not just top seven.
        atomic_torch({'ranks':ranks,'experiment':exp,'version':VERSION,'data_signature':DATA_SIGNATURE},DIR['logs']/f'{exp}_dev_ranks.pt')
        return best
    print('Evaluator ready. Oracle:',ORACLE)

    # -- Notebook source block 24 --
    def training_loss(model,qid,candidates,teacher=None):
        query=batch_tokens([('train','queries',qid)])
        cases=batch_tokens([('train','candidates',c) for c in candidates])
        target=torch.zeros(1,dtype=torch.long,device=DEVICE)
        with amp():
            qvec=model(query);dvec=model(cases)
            scores=(qvec@dvec.T).float()/CFG['temperature']
        ce=F.cross_entropy(scores,target)
        if teacher is None:return ce
        with torch.no_grad(),amp():
            t_scores=(teacher(query)@teacher(cases).T).float()/CFG['temperature']
        T=CFG['kd_temperature']
        kd=F.kl_div(F.log_softmax(scores/T,dim=-1),F.softmax(t_scores/T,dim=-1),reduction='batchmean')*T*T
        return (1-CFG['kd_weight'])*ce+CFG['kd_weight']*kd

    def exp_folder(exp):
        d=DIR['models']/exp;d.mkdir(parents=True,exist_ok=True);return d

    def signature(exp,teacher_exp=None):
        base_info={'version':VERSION,'exp':exp,'cfg':CFG,'spec':EXPS[exp],
           'data':DATA_SIGNATURE,'base_bytes':BASE_PATH.stat().st_size,'teacher_exp':teacher_exp}
        if teacher_exp:
            p=exp_folder(teacher_exp)/'best.pt';st=p.stat()
            base_info['teacher_bytes']=st.st_size
            base_info['teacher_mtime_ns']=st.st_mtime_ns
        return hashlib.sha256(json.dumps(base_info,sort_keys=True).encode()).hexdigest()

    def load_best(exp):
        path=exp_folder(exp)/'best.pt'
        assert path.exists(),path
        state=torch.load(path,map_location='cpu',weights_only=False)
        model=build_experiment(exp)
        model.load_state_dict(state['model'],strict=True)
        return model.to(DEVICE).eval(),state

    def run_experiment(exp,teacher_exp=None):
        spec=EXPS[exp]
        if exp=='E05_KD':assert teacher_exp in ('E01_PARTIAL','E02_IA3','E03_LORMA','E04_FULL')
        path=exp_folder(exp);latest=path/'latest.pt';best=path/'best.pt'
        sig=signature(exp,teacher_exp)
        # Complete-run fast path (no needless model allocation):
        if latest.exists() and best.exists():
            h=torch.load(latest,map_location='cpu',weights_only=False)
            assert h['signature']==sig,'Checkpoint config changed. Preserve old experiments in a different folder.'
            if h.get('done',False):
                b=torch.load(best,map_location='cpu',weights_only=False)
                print(exp,'already complete; best development score:',b['metrics']['micro_f1_at7'])
                return {'experiment':exp,**b['metrics'],'best_weight':b['bm25_weight']}
            del h;gc.collect()
        model=build_experiment(exp).to(DEVICE)
        trainable=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(trainable,lr=spec['lr'],weight_decay=CFG['weight_decay'])
        n_updates_per_epoch=math.ceil(len(TRAIN_LABELLED)/CFG['grad_accum'])
        total_updates=n_updates_per_epoch*CFG['training_epochs']
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max(total_updates,1),eta_min=spec['lr']*.10)
        first_update=0;best_score=-1.0;history=[];evaluated_through=0
        if latest.exists():
            state=torch.load(latest,map_location='cpu',weights_only=False)
            assert state['signature']==sig
            model.load_state_dict(state['model'],strict=True)
            optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
            first_update=state['update'];best_score=state['best_score'];history=state.get('history',[])
            evaluated_through=int(state.get('evaluated_through',0))
            print('RESUME',exp,'from optimizer update',first_update,'/',total_updates)
            del state
        teacher=None
        if teacher_exp:
            teacher,_=load_best(teacher_exp)
            assert sum(p.numel() for p in teacher.parameters())>sum(p.numel() for p in model.parameters())
            teacher.requires_grad_(False).eval()
            print('Teacher:',teacher_exp)
        print('\nTRAIN',exp,params(model),'queries',len(TRAIN_LABELLED),'epochs',CFG['training_epochs'],'optimizer updates',total_updates)
        update_index=0
        for epoch in range(CFG['training_epochs']):
            epoch_end_update=(epoch+1)*n_updates_per_epoch
            # Resume may land exactly after the last update but before validation.
            # Only skip an epoch if its development evaluation was completed.
            if first_update>=epoch_end_update and evaluated_through>=epoch+1:
                print(exp, 'skipping already evaluated epoch', epoch+1)
                continue
            if first_update>epoch_end_update and evaluated_through<epoch+1:
                raise RuntimeError('Checkpoint crossed an epoch whose evaluation is absent.')
            # Absolute optimizer-update index: required when older epochs are skipped.
            update_index=epoch*n_updates_per_epoch
            order=TRAIN_LABELLED.copy()
            random.Random(SEED+epoch).shuffle(order)
            for start in tqdm(range(0,len(order),CFG['grad_accum']),desc=f'{exp} epoch {epoch+1}',leave=True):
                if update_index < first_update:
                    update_index+=1;continue
                group=order[start:start+CFG['grad_accum']]
                model.train();optimizer.zero_grad(set_to_none=True);losses=[]
                for offset,qid in enumerate(group):
                    position=start+offset
                    candidates=training_triplet(qid,epoch,position)
                    loss=training_loss(model,qid,candidates,teacher)
                    if not torch.isfinite(loss):raise RuntimeError(f'{exp} nonfinite loss update {update_index}')
                    (loss/len(group)).backward()
                    losses.append(float(loss.detach()))
                torch.nn.utils.clip_grad_norm_(trainable,1.0)
                optimizer.step();scheduler.step();update_index+=1
                if update_index%CFG['save_every_updates']==0 or update_index==total_updates:
                    atomic_torch({'signature':sig,'update':update_index,'model':model.state_dict(),
                        'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                        'best_score':best_score,'history':history,'evaluated_through':evaluated_through,'done':False},latest)
                    print(f'  checkpoint {exp}: {update_index}/{total_updates}, mean recent loss {np.mean(losses):.4f}')
            # All labeled queries have now contributed once in this epoch.
            dev=validate_dev(model,exp)
            score=float(dev['micro_f1_at7'])
            history.append({'epoch':epoch+1,'update':update_index,'mean_final_batch_loss':np.mean(losses),**dev})
            if score>best_score:
                best_score=score
                atomic_torch({'signature':sig,'model':model.state_dict(),'metrics':dev,
                    'bm25_weight':float(dev['bm25_weight']),'epoch':epoch+1,
                    'source_phase1':str(BASE_PATH),'teacher':teacher_exp,'params':params(model)},best)
                print(exp,'NEW BEST F1:',best_score)
            evaluated_through=epoch+1
            atomic_torch({'signature':sig,'update':update_index,'model':model.state_dict(),
              'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
              'best_score':best_score,'history':history,'evaluated_through':evaluated_through,
              'done':update_index>=total_updates},latest)
            atomic_json(history,DIR['logs']/f'{exp}_train_history.json')
        assert best.exists()
        output={'experiment':exp,'micro_f1_at7':best_score,'best_weight':float(torch.load(best,map_location='cpu',weights_only=False)['bm25_weight'])}
        del model,optimizer,scheduler,teacher;gc.collect();torch.cuda.empty_cache()
        return output
    print('Resumable full-data training engine ready.')

    # -- Notebook source block 26 --
    MAIN_IDS=('E01_PARTIAL','E02_IA3','E03_LORMA','E04_FULL')
    MAIN_RESULTS=[]
    for exp in MAIN_IDS:
        result=run_experiment(exp)
        MAIN_RESULTS.append(result)
        atomic_json(MAIN_RESULTS,DIR['logs']/'main_results.json')
        display(pd.DataFrame(MAIN_RESULTS).sort_values('micro_f1_at7',ascending=False))
    print('Four adaptation experiments complete.')

    # -- Notebook source block 28 --
    TEACHER_ID=max(MAIN_RESULTS,key=lambda r:r['micro_f1_at7'])['experiment']
    print('Selected teacher:',TEACHER_ID)
    KD_RESULT=run_experiment('E05_KD',teacher_exp=TEACHER_ID)
    ALL_RESULTS=MAIN_RESULTS+[KD_RESULT]
    atomic_json({'teacher_id':TEACHER_ID,'experiments':ALL_RESULTS},DIR['logs']/'all_experiments.json')
    display(pd.DataFrame(ALL_RESULTS).sort_values('micro_f1_at7',ascending=False))

    # -- Notebook source block 30 --
    SUBMISSION_MANIFEST=[]
    for exp in EXPS:
        model,best=load_best(exp)
        ranks=neural_ranks(model,'test',exp,cache_test=True)
        predictions=ranked_predictions('test',ranks,float(best['bm25_weight']))
        zippath=submit(predictions,exp)
        rank_path=DIR['cache']/f'{VERSION}_{exp}_{DATA_SIGNATURE}_test_ranks.pt'
        atomic_torch({'experiment':exp,'ranks':ranks,'data_signature':DATA_SIGNATURE},rank_path)
        SUBMISSION_MANIFEST.append({'experiment':exp,'dev_f1':best['metrics']['micro_f1_at7'],
           'bm25_weight':best['bm25_weight'],'zip_path':str(zippath),**best['params']})
        atomic_json(SUBMISSION_MANIFEST,DIR['logs']/'submission_manifest.json')
        print(exp,'done')
        del model,best,ranks,predictions;gc.collect();torch.cuda.empty_cache()
    display(pd.DataFrame(SUBMISSION_MANIFEST).sort_values('dev_f1',ascending=False))

    # -- Notebook source block 32 --
    top2=[r['experiment'] for r in sorted(SUBMISSION_MANIFEST,key=lambda x:-x['dev_f1'])[:2]]
    print('Ensemble models:',top2)

    def saved_ranks(exp,split):
        path=(DIR['logs']/f'{exp}_dev_ranks.pt') if split=='dev' else (DIR['cache']/f'{VERSION}_{exp}_{DATA_SIGNATURE}_test_ranks.pt')
        obj=torch.load(path,map_location='cpu',weights_only=False)
        assert obj['ranks'] and obj.get('data_signature',DATA_SIGNATURE)==DATA_SIGNATURE
        return obj['ranks']

    def ensemble_predictions(split,rank1,rank2,lexical_weight,mix):
        answers={};k=CFG['rrf_k']
        for qid,shortlist in BM25[split].items():
            br=np.arange(1,len(shortlist)+1)
            score=lexical_weight/(k+br)+(1-lexical_weight)*(mix/(k+rank1[qid])+(1-mix)/(k+rank2[qid]))
            ids=np.argsort(-score,kind='stable')[:7]
            answers[qid]=[shortlist[int(i)] for i in ids]
        return answers

    devr1=saved_ranks(top2[0],'dev');devr2=saved_ranks(top2[1],'dev')
    ensemble_grid=[]
    for lexical in (0,.15,.3,.45,.6,.75,.9,1):
        for mix in (.25,.5,.75):
            score=eval_top7(ensemble_predictions('dev',devr1,devr2,lexical,mix))
            ensemble_grid.append({'lexical':lexical,'mix':mix,**score})
    df=pd.DataFrame(ensemble_grid).sort_values('micro_f1_at7',ascending=False,kind='stable')
    display(df.head(12))
    best_ens=df.iloc[0].to_dict();print('Selected ensemble:',best_ens)
    tr1=saved_ranks(top2[0],'test');tr2=saved_ranks(top2[1],'test')
    ensemble=ensemble_predictions('test',tr1,tr2,best_ens['lexical'],best_ens['mix'])
    ensemble_zip=submit(ensemble,'E06_ENSEMBLE')
    atomic_json({'base_models':top2,'selected':best_ens,'all_grid':ensemble_grid,'zip_path':str(ensemble_zip)},DIR['logs']/'ensemble_selection.json')

    # -- Notebook source block 34 --
    EXPECTED=['E00_BM25']+list(EXPS)+['E06_ENSEMBLE']
    validation=[]
    for experiment in EXPECTED:
        z=DIR['submissions']/experiment/'submission.zip'
        assert z.exists(),z
        with zipfile.ZipFile(z) as f:
            assert f.testzip() is None and f.namelist()==['prediction.json']
            data=json.loads(f.read('prediction.json'))
        assert set(data)==set(DATA['test']['queries'])
        for qid,ids in data.items():
            assert len(ids)==7 and len(set(ids))==7,(experiment,qid)
            assert all(isinstance(cid,str) and cid in DATA['test']['candidates'] for cid in ids),(experiment,qid)
        validation.append({'experiment':experiment,'query_count':len(data),'valid':True,'zip':str(z)})
    display(pd.DataFrame(validation))
    summary={'version':VERSION,'phase1_checkpoint':str(BASE_PATH),
             'training_queries_all':len(TRAIN_ALL),'training_queries_labeled':len(TRAIN_LABELLED),
             'training_query_cap':None,'training_epochs':CFG['training_epochs'],
             'config':CFG,'bm25':BM25_METRICS,'shortlist_oracle':ORACLE,
             'teacher':TEACHER_ID,'neural_experiments':SUBMISSION_MANIFEST,
             'ensemble':best_ens,'submission_validation':validation}
    atomic_json(summary,DIR['logs']/'FINAL_PHASE7_SUMMARY.json')
    print('\nALL SUBMISSION ZIPs VALIDATED. Folder:',DIR['submissions'])
    print('Course requirement: submit multiple distinct methods, including LORMA+ and KD.')
    print('Best dev neural:',max(SUBMISSION_MANIFEST,key=lambda r:r['dev_f1']))
    print('Best dev ensemble:',best_ens['micro_f1_at7'],'BM25:',BM25_METRICS['micro_f1_at7'])
    print('Choose the final leaderboard entry using development/actual Codabench outcomes; do not presume ensemble wins.')

if __name__ == "__main__":
    main()
