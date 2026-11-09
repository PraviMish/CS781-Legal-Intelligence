"""Prior Case Retrieval: historically reproduced E07-E10 V2 ranking pipeline

Faithful refactoring of executed CS781 Capstone Phase 7 notebook code.
Original experimental algorithms and hyperparameters are preserved.
CLI replaces Colab Drive mounting and notebook class restoration.
"""
from __future__ import annotations
import argparse

from phase7.validation import split_directory

def main(argv=None):
    # -- Notebook source block 4 --
    import os, sys, re, gc, json, time, math, hashlib, pickle, zipfile, random
    from pathlib import Path
    from collections import defaultdict
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, TfidfTransformer
    from IPython.display import display
    from tqdm.auto import tqdm

    SEED=230787
    random.seed(SEED)
    np.random.seed(SEED)
    parser=argparse.ArgumentParser(description="Reproduce Phase 7 E07--E10 (TF-IDF, citations, LightGBM LambdaRank and fusion)")
    parser.add_argument('--project-root',type=Path,default=Path.cwd(),help='Folder containing phase7/; defaults to current directory')
    parser.add_argument('--data-dir',type=Path,help='Folder with train/,dev/,test/; defaults to project-root/phase7/data')
    parser.add_argument('--cache-dir',type=Path,help='Optional cache directory')
    parser.add_argument('--output-dir',type=Path,help='Optional submission output directory')
    parser.add_argument('--log-dir',type=Path,help='Optional log/model directory')
    args=parser.parse_args(argv)
    ROOT=args.project_root.expanduser().resolve()
    P7=ROOT/'phase7'
    DATA_DIR=args.data_dir.expanduser().resolve() if args.data_dir else P7/'data'
    CACHE=args.cache_dir.expanduser().resolve() if args.cache_dir else P7/'cache'/'v2_retrieval_fix'
    OUT=args.output_dir.expanduser().resolve() if args.output_dir else P7/'submissions'/'V2_RETRIEVAL_FIX'
    LOG=args.log_dir.expanduser().resolve() if args.log_dir else P7/'logs'/'v2_retrieval_fix'
    for p in (CACHE,OUT,LOG):p.mkdir(parents=True,exist_ok=True)
    VERSION='p7_retrieval_fix_v2_20261107'
    SPLITS=('train','dev','test')
    # Memory/computational safety; these are *not* training-query caps.
    LEX_MAX_FEATURES=180_000
    HASH_DIM=2**18
    RANKER_TOP_HARD=200
    RANKER_TOP_UNI=100
    RANKER_RANDOM=48
    N_TREES=160


    def atomic_json(obj,path):
        path=Path(path);tmp=path.with_name(path.name+'.tmp')
        with open(tmp,'w',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
        os.replace(tmp,path)

    def atomic_pickle(obj,path):
        path=Path(path);tmp=path.with_name(path.name+'.tmp')
        with open(tmp,'wb') as f:pickle.dump(obj,f,protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp,path)

    def load_pickle(path):
        with open(path,'rb') as f:return pickle.load(f)

    print('ROOT',ROOT);print('VERSION',VERSION)
    print('Outputs',OUT)
    print('No GPU is required for this notebook.')

    # -- Notebook source block 6 --
    def load_split(split):
        assert split in SPLITS
        directory = split_directory(DATA_DIR, split)
        qfile = directory / f'{split}.json'
        cfile = directory / f'{split}_candidates.json'
        for p in (qfile,cfile):
            assert p.is_file(),('Missing official dataset file',p)
        with open(qfile,encoding='utf-8-sig') as f:Q=json.load(f)
        with open(cfile,encoding='utf-8-sig') as f:C=json.load(f)
        assert isinstance(Q,dict) and isinstance(C,dict)
        assert len(Q)>0 and len(C)>=7
        qids=list(Q);cids=list(C)
        # Preserve exact ID strings, including leading zeros.
        assert all(isinstance(x,str) for x in qids+cids)
        gold={qid:set(str(x) for x in row.get('relevant_candidates',[])) for qid,row in Q.items()}
        if split!='test':
            assert all(g.issubset(C) for g in gold.values()),'Gold IDs must belong to this split candidate pool'
        else:
            assert all(not g for g in gold.values()),'Test labels must remain private'
        texts_q=[' '.join(row['text']) if isinstance(row['text'],list) else str(row['text']) for row in Q.values()]
        texts_c=[' '.join(row['text']) if isinstance(row['text'],list) else str(row['text']) for row in C.values()]
        assert all(t.strip() for t in texts_q+texts_c)
        return {'query_ids':qids,'candidate_ids':cids,'query_texts':texts_q,'candidate_texts':texts_c,'gold':gold}

    # Run the small official development split immediately.
    DEV=load_split('dev')
    print('DEV:',len(DEV['query_ids']),'queries,',len(DEV['candidate_ids']),'candidate cases')
    print('Gold references:',sum(map(len,DEV['gold'].values())))
    assert len(DEV['query_ids'])==118 and len(DEV['candidate_ids'])==1023, 'Dataset layout differs; inspect before proceeding.'

    def metric(S,split_data):
        qids=split_data['query_ids'];cids=split_data['candidate_ids'];gold=split_data['gold']
        assert S.shape==(len(qids),len(cids)),(S.shape,len(qids),len(cids))
        total_gold=sum(map(len,gold.values()))
        assert total_gold>0,'Metrics unavailable on unlabeled test.'
        idx=np.argsort(-S,axis=1,kind='stable')[:,:7]
        hits=sum(len(gold[qid]&{cids[int(j)] for j in idx[i]}) for i,qid in enumerate(qids))
        return {'micro_f1_at7':2*hits/(7*len(qids)+total_gold),
                'micro_precision_at7':hits/(7*len(qids)),
                'micro_recall_at7':hits/total_gold,
                'hits':int(hits),'gold_pairs':int(total_gold)}

    def build_predictions(S,split_data):
        idx=np.argsort(-S,axis=1,kind='stable')[:,:7]
        return {qid:[split_data['candidate_ids'][int(j)] for j in idx[i]]
                for i,qid in enumerate(split_data['query_ids'])}

    def submit(predictions,slug,split_data):
        assert set(predictions)==set(split_data['query_ids'])
        allowed=set(split_data['candidate_ids'])
        for qid,ids in predictions.items():
            assert len(ids)==7 and len(set(ids))==7,qid
            assert all(isinstance(x,str) and x in allowed for x in ids),qid
        folder=OUT/slug;folder.mkdir(exist_ok=True)
        json_path=folder/'prediction.json';zip_path=folder/'submission.zip'
        atomic_json(predictions,json_path)
        with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:z.write(json_path,arcname='prediction.json')
        with zipfile.ZipFile(zip_path) as z:
            assert z.namelist()==['prediction.json'] and z.testzip() is None
            assert json.loads(z.read('prediction.json'))==predictions
        print('VALID',slug,'->',zip_path,'queries',len(predictions))
        return zip_path

    # -- Notebook source block 8 --
    # These fields are present in the PUBLIC document text itself. They are not ground-truth labels.
    def section(s,start,end,limit=25000):
        p=s[:limit];a=re.search(start,p,re.IGNORECASE)
        if not a:return ''
        b=re.search(end,p[a.end():],re.IGNORECASE)
        return p[a.end():a.end()+b.start()] if b else p[a.end():]

    def official_air_citation(s):
        header=section(s,r'\bCITATION\s*:',r'CITATOR\s+INFO|ACT:|HEADNOTE:',limit=7000)
        m=re.search(r'\b(19\d\d|20\d\d)\s+AIR\s+(\d+)\b',header,re.IGNORECASE)
        return m.groups() if m else None

    def sc_citations(s):
        return set(re.findall(r'\b(19\d\d|20\d\d)\s+SC\s*(\d{2,5})\b',s,re.IGNORECASE))

    def citation_matrices(d):
        cache=CACHE/f'{VERSION}_citation_{len(d["query_ids"])}_{len(d["candidate_ids"])}.npz'
        if cache.exists():
            a=np.load(cache)
            assert a['incoming'].shape==(len(d['query_ids']),len(d['candidate_ids']))
            return a['incoming'],a['mentioned']
        # Incoming: query's citation is indexed in a candidate's CITATOR INFO field.
        # Mentioned: candidate's AIR citation is explicitly mentioned inside the query.
        candidate_air=[official_air_citation(s) for s in d['candidate_texts']]
        candidate_incoming=[sc_citations(section(s,r'CITATOR\s+INFO\s*:',r'ACT:|HEADNOTE:|JUDGMENT'))
                            for s in d['candidate_texts']]
        query_air=[official_air_citation(s) for s in d['query_texts']]
        query_mentions=[sc_citations(s) for s in d['query_texts']]
        incoming=np.zeros((len(d['query_ids']),len(d['candidate_ids'])),dtype=np.uint8)
        mentioned=np.zeros_like(incoming)
        for i,(a,matches) in enumerate(zip(query_air,query_mentions)):
            for j,(citation,refs) in enumerate(zip(candidate_air,candidate_incoming)):
                incoming[i,j]=int(a is not None and a in refs)
                mentioned[i,j]=int(citation is not None and citation in matches)
        with open(str(cache)+'.tmp','wb') as f:np.savez_compressed(f,incoming=incoming,mentioned=mentioned)
        os.replace(str(cache)+'.tmp',cache)
        return incoming,mentioned

    DEV_INC,DEV_MENTION=citation_matrices(DEV)
    A=DEV_INC.astype(bool);B=DEV_MENTION.astype(bool)
    y=np.array([[int(cid in DEV['gold'][qid]) for cid in DEV['candidate_ids']]
                for qid in DEV['query_ids']],dtype=bool)
    print('Incoming citation matches: positives',int((A&y).sum()),'negatives',int((A&~y).sum()))
    print('Direct citation matches: positives',int((B&y).sum()),'negatives',int((B&~y).sum()))

    # -- Notebook source block 10 --
    def lexical_tfidf(d):
        """Split-local TF-IDF. Each split indexes *only its own* public candidate pool.
           Fitting TF-IDF uses candidate text only, never dev or test gold labels.
        """
        cache=CACHE/f'{VERSION}_lexical_{len(d["query_ids"])}_{len(d["candidate_ids"])}.npy'
        shape=(len(d['query_ids']),len(d['candidate_ids']))
        if cache.exists():
            x=np.load(cache)
            assert x.shape==shape
            return x
        v=TfidfVectorizer(lowercase=True,stop_words='english',ngram_range=(1,2),
                          min_df=2,max_features=LEX_MAX_FEATURES,sublinear_tf=True,
                          norm='l2',dtype=np.float32)
        X=v.fit_transform(d['candidate_texts'])
        Y=v.transform(d['query_texts'])
        S=(Y@X.T).toarray().astype(np.float32)
        del X,Y,v;gc.collect()
        temp=str(cache)+'.tmp'
        with open(temp,'wb') as f:np.save(f,S)
        os.replace(temp,cache)
        return S

    DEV_LEX=lexical_tfidf(DEV)
    print('Raw full-document TF-IDF:',metric(DEV_LEX,DEV))
    assert metric(DEV_LEX,DEV)['micro_f1_at7']>.35,'Expected test score not reproduced: inspect package versions/data'

    # Fixed, compact development selection grid.
    WEIGHTS=[(a,b) for a in (0.1,0.4,1.5) for b in (0.0,0.02,0.08)]
    CITE_GRID=[]
    for a,b in WEIGHTS:
        m=metric(DEV_LEX+a*DEV_INC+b*DEV_MENTION,DEV)
        CITE_GRID.append({'incoming_weight':a,'mention_weight':b,**m})
    GRID=pd.DataFrame(CITE_GRID).sort_values(['micro_f1_at7','hits'],ascending=False,kind='stable')
    display(GRID)
    SELECTED=GRID.iloc[0].to_dict()
    print('Chosen citation boosts:',SELECTED['incoming_weight'],SELECTED['mention_weight'])
    DEV_LEGAL=DEV_LEX+SELECTED['incoming_weight']*DEV_INC+SELECTED['mention_weight']*DEV_MENTION
    print('Citation-aware legal TF-IDF:',metric(DEV_LEGAL,DEV))
    atomic_json({'grid':CITE_GRID,'selected':SELECTED},LOG/'lexical_development.json')

    # -- Notebook source block 12 --
    TEST=load_split('test')
    TEST_INC,TEST_MENTION=citation_matrices(TEST)
    TEST_LEX=lexical_tfidf(TEST)
    print('Test queries:',len(TEST['query_ids']),'candidate pool:',len(TEST['candidate_ids']))
    print('Zero test labels:',all(not x for x in TEST['gold'].values()))
    submit(build_predictions(TEST_LEX,TEST),'E07_TFIDF',TEST)
    TEST_LEGAL=TEST_LEX+SELECTED['incoming_weight']*TEST_INC+SELECTED['mention_weight']*TEST_MENTION
    submit(build_predictions(TEST_LEGAL,TEST),'E08_LEGAL_CITATIONS',TEST)
    print('Two improved submissions are already available, even without the optional ranker.')

    # -- Notebook source block 14 --
    def get_old_bm25_ranks(split,d):
        # Old Phase 7 BM25 is included only as an optional feature.
        paths=sorted((P7/'cache').glob(f'p7_final_v1_*_bm25_{split}_256.pkl'))
        out=np.full((len(d['query_ids']),len(d['candidate_ids'])),9999,dtype=np.int32)
        if not paths:return out
        try:
            ranks=load_pickle(paths[-1]);cmap={cid:i for i,cid in enumerate(d['candidate_ids'])}
            if set(ranks)!=set(d['query_ids']):return out
            for i,qid in enumerate(d['query_ids']):
                for k,cid in enumerate(ranks[qid],start=1):
                    if cid in cmap:out[i,cmap[cid]]=k
            print('Loaded previous BM25 rankings:',split)
        except Exception as exc:
            print('Ignoring incompatible BM25 cache:',exc)
        return out

    def hashed_tfidf(d,ngram=(1,2)):
        # Memory-bounded vocabulary for the large 160 MB training candidate file.
        hasher=HashingVectorizer(n_features=HASH_DIM,alternate_sign=False,
               lowercase=True,stop_words='english',ngram_range=ngram,
               norm=None,dtype=np.float32)
        X=hasher.transform(d['candidate_texts'])
        Y=hasher.transform(d['query_texts'])
        tfidf=TfidfTransformer(sublinear_tf=True,norm='l2')
        X=tfidf.fit_transform(X)
        Y=tfidf.transform(Y)
        score=(Y@X.T).toarray().astype(np.float32)
        del X,Y,hasher,tfidf;gc.collect()
        return score

    def ranking_features(split,d,inc,mentioned):
        cache=CACHE/f'{VERSION}_features_{split}_{len(d["query_ids"])}_{len(d["candidate_ids"])}.npy'
        shape=(len(d['query_ids']),len(d['candidate_ids']),8)
        if cache.exists():
            F=np.load(cache,mmap_mode='r')
            assert F.shape==shape
            return F
        full=hashed_tfidf(d,(1,2))
        print(split,'full-document hashed cosine computed')
        uni=hashed_tfidf(d,(1,1))
        print(split,'unigram hashed cosine computed')
        bm=get_old_bm25_ranks(split,d)
        qlens=np.maximum(1,np.array([len(x) for x in d['query_texts']],dtype=np.float32))
        clens=np.maximum(1,np.array([len(x) for x in d['candidate_texts']],dtype=np.float32))
        # Relative rank (query-level normalized) and candidate/query length ratio.
        order=np.argsort(-full,axis=1,kind='stable')
        ranks=np.empty_like(order,dtype=np.int32)
        np.put_along_axis(ranks,order,np.arange(1,len(d['candidate_ids'])+1)[None,:],axis=1)
        F=np.stack([
            full,uni,inc.astype(np.float32),mentioned.astype(np.float32),
            1/(50+bm.astype(np.float32)),
            1/np.sqrt(ranks.astype(np.float32)),
            np.log1p(clens[None,:]/qlens[:,None])*np.ones_like(full),
            np.log1p(qlens[:,None]/clens[None,:])*np.ones_like(full)
        ],axis=-1).astype(np.float32)
        tmp=str(cache)+'.tmp'
        with open(tmp,'wb') as f:np.save(f,F)
        os.replace(tmp,cache)
        del full,uni,bm,ranks;gc.collect()
        return np.load(cache,mmap_mode='r')

    print('Feature construction is ready; all candidate pools remain strictly split-local.')

    # -- Notebook source block 16 --
    MODEL_PATH=LOG/'lambda_ranker_all_positives.pkl'
    META_PATH=LOG/'lambda_ranker_config.json'
    if MODEL_PATH.exists() and META_PATH.exists():
        META=json.load(open(META_PATH))
        assert META['version']==VERSION
        ranker=load_pickle(MODEL_PATH)
        print('Reusing trained legal LambdaRank model from Drive.')
    else:
        TRAIN=load_split('train')
        TRAIN_INC,TRAIN_MENTION=citation_matrices(TRAIN)
        TRAIN_F=ranking_features('train',TRAIN,TRAIN_INC,TRAIN_MENTION)
        CIDS=TRAIN['candidate_ids']
        CID_TO_IDX={cid:i for i,cid in enumerate(CIDS)}
        print('Training with every labeled query; all positives must be included.')
        rng=np.random.default_rng(SEED)
        arrays=[];labels=[];groups=[];used_positive=0;skipped_queries=0
        for qi,qid in enumerate(tqdm(TRAIN['query_ids'],desc='Constructing candidate ranking groups')):
            gold=TRAIN['gold'][qid]
            if not gold:
                skipped_queries+=1;continue
            positive=np.array([CID_TO_IDX[x] for x in gold],dtype=np.int32)
            row=TRAIN_F[qi]
            h1=np.argsort(-row[:,0],kind='stable')[:RANKER_TOP_HARD]
            h2=np.argsort(-row[:,1],kind='stable')[:RANKER_TOP_UNI]
            citations=np.flatnonzero((row[:,2]>0)|(row[:,3]>0))
            rand=rng.choice(len(CIDS),size=min(RANKER_RANDOM,len(CIDS)),replace=False)
            indices=np.unique(np.concatenate([positive,h1,h2,citations,rand]))
            # Query-group order and labels are aligned by construction.
            x=np.asarray(row[indices],dtype=np.float32)
            y=np.isin(indices,positive).astype(np.int32)
            assert y.sum()==len(positive),(qid,y.sum(),len(positive))
            used_positive+=len(positive)
            arrays.append(x);labels.append(y);groups.append(len(indices))
        assert used_positive==sum(len(g) for g in TRAIN['gold'].values())
        assert skipped_queries==10,('unexpected count of unlabeled training queries',skipped_queries)
        X=np.concatenate(arrays);y=np.concatenate(labels);del arrays,labels,TRAIN_F;gc.collect()
        print('Ranker training examples:',len(y),'positive pairs:',int(y.sum()),'queries:',len(groups))
        assert int(y.sum())==used_positive==5308
        ranker=lgb.LGBMRanker(
            objective='lambdarank',metric='ndcg',n_estimators=N_TREES,
            learning_rate=0.055,num_leaves=15,min_child_samples=65,
            reg_lambda=5.0,colsample_bytree=0.9,
            max_bin=127,random_state=SEED,n_jobs=4,
            verbosity=-1,lambdarank_truncation_level=10,
            label_gain=[0,1]
        )
        ranker.fit(X,y,group=groups,eval_at=[7]);del X,y;gc.collect()
        atomic_pickle(ranker,MODEL_PATH)
        META={'version':VERSION,'groups':len(groups),'all_gold_pairs':used_positive,
              'hard_candidates':RANKER_TOP_HARD,'unigram_candidates':RANKER_TOP_UNI,
              'random_candidates':RANKER_RANDOM,'n_trees':N_TREES}
        atomic_json(META,META_PATH)
        del TRAIN,TRAIN_INC,TRAIN_MENTION
        gc.collect()
    print('LambdaRank model available:',MODEL_PATH)
    print('Feature importances:',ranker.feature_importances_)

    # -- Notebook source block 18 --
    DEV_F=ranking_features('dev',DEV,DEV_INC,DEV_MENTION)

    def ranker_score(features):
        n,k,f=features.shape
        scores=np.empty((n,k),dtype=np.float32)
        # Predict per query to bound peak RAM for large test pools.
        for i in tqdm(range(n),desc='LightGBM scoring'):
            scores[i]=ranker.predict(np.asarray(features[i],dtype=np.float32))
        return scores

    DEV_RANK=ranker_score(DEV_F)
    print('LightGBM only:',metric(DEV_RANK,DEV))
    print('Legal TF-IDF + citations:',metric(DEV_LEGAL,DEV))

    # Rank fusion is more robust across heterogeneous score scales.
    def reciprocal_ranks(scores,k=30):
        order=np.argsort(-scores,axis=1,kind='stable')
        rank=np.empty_like(order,dtype=np.int32)
        np.put_along_axis(rank,order,np.arange(1,scores.shape[1]+1)[None,:],axis=1)
        return (1.0/(k+rank)).astype(np.float32)

    DEV_A=reciprocal_ranks(DEV_LEGAL)
    DEV_B=reciprocal_ranks(DEV_RANK)
    WEIGHT_OPTIONS=[0.0,0.2,0.4,0.6,0.8,1.0]
    MIXES=[]
    for a in WEIGHT_OPTIONS:
        score=a*DEV_A+(1-a)*DEV_B
        MIXES.append({'legal_lexical_weight':a,**metric(score,DEV)})
    MIX_DF=pd.DataFrame(MIXES).sort_values('micro_f1_at7',ascending=False,kind='stable')
    display(MIX_DF)
    BEST_MIX=MIX_DF.iloc[0].to_dict()
    print('Development-selected best mix:',BEST_MIX)
    atomic_json({'mix_grid':MIXES,'selected':BEST_MIX},LOG/'lambda_rank_dev_results.json')

    # -- Notebook source block 20 --
    TEST_F=ranking_features('test',TEST,TEST_INC,TEST_MENTION)
    TEST_RANK=ranker_score(TEST_F)
    submit(build_predictions(TEST_RANK,TEST),'E09_LIGHTGBM',TEST)
    TEST_A=reciprocal_ranks(TEST_LEGAL)
    TEST_B=reciprocal_ranks(TEST_RANK)
    a=float(BEST_MIX['legal_lexical_weight'])
    TEST_BEST=a*TEST_A+(1-a)*TEST_B
    best_name='E10_DEV_SELECTED_LEGAL_RANKER'
    final_path=submit(build_predictions(TEST_BEST,TEST),best_name,TEST)

    # One unified directory contains ALL improved ZIPs.
    manifest={'version':VERSION,
              'old_phase7_models_preserved':True,
              'lexical_dev':metric(DEV_LEX,DEV),
              'citation_aware_dev':metric(DEV_LEGAL,DEV),
              'ranker_dev':metric(DEV_RANK,DEV),
              'final_dev':metric(float(BEST_MIX['legal_lexical_weight'])*DEV_A+
                                  (1-float(BEST_MIX['legal_lexical_weight']))*DEV_B,DEV),
              'final_submission':str(final_path),
              'all_improved_submissions':[str(p) for p in sorted(OUT.glob('*/submission.zip'))],
              'course_note':'Original four adapted models and KD remain in phase7/models and phase7/submissions'
    }
    atomic_json(manifest,LOG/'FINAL_MANIFEST.json')
    display(pd.DataFrame([{'submission':p.parent.name,'zip':str(p),'size_bytes':p.stat().st_size}
                          for p in sorted(OUT.glob('*/submission.zip'))]))
    print('NEW FINAL ZIP:',final_path)
    print('Existing Phase 7 checkpoints were not touched.')

if __name__ == "__main__":
    main()
