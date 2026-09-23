#!/usr/bin/env python
"""R0 cache builder and preregistered R1 lightweight temporal probe."""
from __future__ import annotations
import argparse, csv, hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.research.temporal_rep.core import SmallTokenTCN, robust_channel_stats, topk_mean
from scripts.research.tier1_challenger import gyro5_feature_block

OUT=ROOT/'outputs/challenger_70/temporal_rep'; CACHE=OUT/'cache'; R1=OUT/'r1_token_probe_002'
SEED=20260923; MAX_TOKENS=47; TOKEN_SAMPLES=50

def dump(p,v): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n',encoding='utf8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def population():
    p=ROOT/'outputs/challenger_70/e04_verifier_audit'; frames=[]
    for name,file,label in [('matched_tp','tp_reference.csv',1),('final_fp','fp_reference.csv',0),('reachable_fn','reachable_fn_candidates.csv',1)]:
        d=pd.read_csv(p/file)
        if name=='reachable_fn': d=d.sort_values(['truth_key','blend_score'],ascending=[True,False]).drop_duplicates('truth_key')
        d=d.assign(population=name,label=label); frames.append(d)
    return pd.concat(frames,ignore_index=True,sort=False)

def slice_tokens(row):
    with np.load(ROOT/'cache/sessions'/f"{row.session_id}.npz",allow_pickle=False) as z:
        rate=float(z['row_rate']); step=max(1,int(round(rate/10))); acc=z['acc'][:,::step]; gyro=z['gyro'][:,::step]; ts=z['t_acc'][::step]; valid=z['imu_valid'][::step]
    centre=int((row.start_ms+row.end_ms)//2); span=min(120_000,max(5_000,int(row.end_ms-row.start_ms))); first=centre-span//2+2500; last=centre+span//2-2500
    centers=np.arange(first,last+1,2500,dtype=np.int64)[:MAX_TOKENS]; tokens=[]
    for c in centers:
        m=(ts>=c-2500)&(ts<=c+2500)&valid
        if m.sum()<4: continue
        x=np.vstack([acc[:,m],gyro[:,m]]).astype(np.float32); old=np.linspace(0,1,x.shape[1]); new=np.linspace(0,1,TOKEN_SAMPLES)
        tokens.append(np.vstack([np.interp(new,old,ch) for ch in x]))
    return np.asarray(tokens,np.float32)

def r0():
    if CACHE.exists(): raise RuntimeError('R0 cache already exists')
    d=population(); arrays=np.zeros((len(d),MAX_TOKENS,6,TOKEN_SAMPLES),np.float32); mask=np.zeros((len(d),MAX_TOKENS),bool)
    started=time.perf_counter()
    for i,row in d.iterrows():
        x=slice_tokens(row); arrays[i,:len(x)]=x; mask[i,:len(x)]=True
    CACHE.mkdir(parents=True); np.savez_compressed(CACHE/'candidate_tokens.npz',tokens=arrays,mask=mask,labels=d.label.to_numpy(np.int8),folds=d.fold.to_numpy(np.int8),subjects=d.subject_id.to_numpy(str),populations=d.population.to_numpy(str),candidate_keys=d.candidate_key.to_numpy(str))
    d[['candidate_key','session_id','subject_id','fold','population','label','start_ms','end_ms']].to_csv(CACHE/'candidate_index.csv',index=False)
    manifest={'status':'R0_READY','rows':len(d),'population_counts':d.population.value_counts().to_dict(),'token_seconds':5,'stride_seconds':2.5,'max_candidate_span_seconds':120,'token_samples':50,'channels':['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z'],'gravity_alignment':'source cache coordinates; no additional learned transform','fd_preprocessed':(ROOT/'FDdatasets/doi-10.48804-CN8VBB/v1.0/FD-I/X_L.pkl').exists(),'fd_checkpoint':(ROOT/'checkpoints/fd_pretrained_s1.pt').exists(),'runtime_seconds':time.perf_counter()-started}
    dump(CACHE/'manifest.json',manifest); print(json.dumps(manifest))

def prereg():
    if R1.exists(): raise RuntimeError('R1 already exists')
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    plan={'experiment':'R1-002','derived_from':'R1-001 implementation failure','diagnostic_only':True,'code_sha':code,'architecture':'SmallTokenTCN','channels':24,'blocks':3,'dilations':[1,2,4],'embedding_dim':16,'parameter_limit':100000,'token_seconds':5,'stride_seconds':2.5,'candidate_span_seconds':120,'pooling':'top-k mean','topk_fraction':.2,'epochs':30,'seed':SEED,'device':'cuda if available','normalization':'fold-contained channel median/MAD','populations':{'positive':['matched_tp','reachable_fn'],'negative':['final_fp']},'controls':['GYRO5 scalar logistic','frozen random TCN embedding logistic'],'challenger':'trained TCN embedding + GYRO5 scalar logistic','probe_C':1.0,'gate':{'trained_minus_random_f1':.02,'incremental_minus_scalar_f1':.02,'direction_folds':4,'reachable_fn_tp_like':True},'outer_f1':False,'abort':['subject overlap','cache mismatch','parameter count >=100000']}
    dump(R1/'plan.json',plan)

def bag_embedding(model,x,mask):
    n,b,c,t=x.shape; scores,emb=model(x.reshape(n*b,c,t)); scores=scores.reshape(n,b); emb=emb.reshape(n,b,-1); out=[]
    for s,e,m in zip(scores,emb,mask):
        idx=torch.where(m)[0]; k=max(1,int(np.ceil(len(idx)*.2))); chosen=idx[torch.topk(s[idx],k).indices]; out.append(e[chosen].mean(0))
    return torch.stack(out),topk_mean(scores,mask,.2)

def run():
    plan=json.loads((R1/'plan.json').read_text()); z=np.load(CACHE/'candidate_tokens.npz'); X=z['tokens']; mask=z['mask']; y=z['labels']; folds=z['folds']; pops=z['populations']; subjects=z['subjects']
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); rows=[]; all_pred={k:np.zeros(len(y)) for k in ['scalar','random','trained']}; started=time.perf_counter()
    for fold in range(5):
        tr=folds!=fold; va=~tr
        if set(subjects[tr])&set(subjects[va]): raise RuntimeError('subject leakage')
        center,scale=robust_channel_stats(X[tr],mask[tr]); xn=(X-center[None,None,:,None])/scale[None,None,:,None]
        xt=torch.tensor(xn,dtype=torch.float32,device=device); mt=torch.tensor(mask,device=device); yt=torch.tensor(y,dtype=torch.float32,device=device)
        torch.manual_seed(SEED+fold); model=SmallTokenTCN().to(device); optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
        train_idx=torch.tensor(np.flatnonzero(tr),device=device)
        for _ in range(30):
            model.train(); optimizer.zero_grad(); _,score=bag_embedding(model,xt[train_idx],mt[train_idx]); loss=torch.nn.functional.binary_cross_entropy_with_logits(score,yt[train_idx]); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): trained,_=bag_embedding(model,xt,mt)
        torch.manual_seed(SEED+100+fold); random=SmallTokenTCN().to(device).eval()
        with torch.no_grad(): rand,_=bag_embedding(random,xt,mt)
        scalar=np.vstack([gyro5_feature_block(X[i,mask[i],3:,:].transpose(1,0,2).reshape(3,-1),10) for i in range(len(y))])
        features={'scalar':scalar,'random':np.column_stack([scalar,rand.cpu().numpy()]),'trained':np.column_stack([scalar,trained.cpu().numpy()])}
        foldrow={'fold':fold}
        for name,f in features.items():
            probe=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=SEED))
            probe.fit(f[tr],y[tr]); pred=probe.predict_proba(f[va])[:,1]; all_pred[name][va]=pred
            foldrow[name]={'f1':f1_score(y[va],pred>=.5),'auc':roc_auc_score(y[va],pred),'tp_fp_direction':float(np.median(pred[(va)&(pops=='matched_tp')])-np.median(pred[(va)&(pops=='final_fp')]))}
        rows.append(foldrow)
    agg={}
    for name,pred in all_pred.items(): agg[name]={'f1':f1_score(y,pred>=.5),'auc':roc_auc_score(y,pred),'matched_tp_median':float(np.median(pred[pops=='matched_tp'])),'reachable_fn_median':float(np.median(pred[pops=='reachable_fn'])),'final_fp_median':float(np.median(pred[pops=='final_fp']))}
    direction=sum(r['trained']['tp_fp_direction']>0 for r in rows); fn_like=abs(agg['trained']['reachable_fn_median']-agg['trained']['matched_tp_median'])<abs(agg['trained']['reachable_fn_median']-agg['trained']['final_fp_median'])
    passed=agg['trained']['f1']>=agg['random']['f1']+.02 and agg['trained']['f1']>=agg['scalar']['f1']+.02 and direction>=4 and fn_like
    status='TEMPORAL_REPRESENTATION_SUPPORTED' if passed else 'TEMPORAL_REPRESENTATION_NOT_SUPPORTED'
    dump(R1/'fold_metrics.json',rows); dump(R1/'aggregate_metrics.json',agg); dump(R1/'inner_gate.json',{'status':status,'direction_folds':direction,'reachable_fn_tp_like':fn_like,'outer_f1':False}); dump(R1/'runtime.json',{'seconds':time.perf_counter()-started,'device':str(device),'parameters':sum(p.numel() for p in SmallTokenTCN().parameters())})
    dump(R1/'manifest.json',{'status':status,'cache_sha256':sha(CACHE/'candidate_tokens.npz'),'subject_isolation':True,'fold_contained_normalization':True,'outer_f1':False})
    (R1/'report.md').write_text(f"# R1 temporal token probe\n\nStatus: `{status}`\n\nScalar F1 `{agg['scalar']['f1']:.4f}`, random encoder F1 `{agg['random']['f1']:.4f}`, trained encoder incremental F1 `{agg['trained']['f1']:.4f}`. Direction folds `{direction}/5`; reachable FN TP-like `{fn_like}`.\n",encoding='utf8')
    print(json.dumps({'status':status,'aggregate':agg,'direction_folds':direction}))

if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('--phase',choices=['r0','prereg','run'],required=True); x=a.parse_args(); {'r0':r0,'prereg':prereg,'run':run}[x.phase]()
