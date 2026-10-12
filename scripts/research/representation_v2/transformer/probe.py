"""Sealed A6 T0/T1 execution. No outer or release entry point exists."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import gc
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from scripts.research.representation_v2 import event_probe as ep
from scripts.research.representation_v2.provenance import assert_no_heldout_ancestors
from src.pipeline.challenger.e03 import _e03_outer_subject_ids
from .cache import audit_data, file_hash, encoder_hash, load_or_encode
from .contract import ARMS, IDENTITY, OUTPUT, SOURCE_SHA, SOURCE_TARGET_EVIDENCE, T1Config, digest, validate_identities, validate_fit_counts
from .timeline import candidate_chunks
from .training import fit_head, predict_head, fuse_scores

SPEC='docs/challenge_070_amendment_a6_transformer_design.md'
PLAN='docs/superpowers/plans/2026-10-12-a6-transformer-t0-t1-plan.md'


def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    try:
        tmp.write_text(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
        tmp.replace(path)
    finally: tmp.unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@contextmanager
def phase(output,expected,success):
    output=Path(output); state=read_json(output/'state.json')
    if state['status']!=expected: raise RuntimeError('unexpected attempt state; no overwrite/retry')
    write_json(output/'state.json',{**state,'status':success+'_RUNNING'})
    try:
        yield
    except BaseException as exc:
        write_json(output/'state.json',{**state,'status':'INCOMPLETE','error':repr(exc),'outer_calls':0})
        raise
    else:
        write_json(output/'state.json',{**state,'status':success,'outer_calls':0})


def verify_registered_files(root,plan):
    for name,sha in plan['code_files'].items():
        if not (Path(root)/name).is_file() or file_hash(Path(root)/name)!=sha:
            raise RuntimeError('registered file changed: '+name)


def require_clean(root):
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
        raise RuntimeError('commit code and evidence before next stage')


def require_committed(root,path):
    relative=Path(path).relative_to(root).as_posix()
    committed=subprocess.check_output(['git','rev-parse','HEAD:'+relative],cwd=root,text=True).strip()
    working=subprocess.check_output(['git','hash-object','--path='+relative,str(path)],cwd=root,text=True).strip()
    if committed!=working: raise RuntimeError('artifact must be committed: '+relative)


def build_manifest(root):
    config=ep.registered_e03_config(0,workers=4)
    if config.inner_splits!=4: raise ValueError('registered producer OOF count differs')
    payload=ep.load_source_artifact(Path(root)/'outputs/challenger_70/representation_v2/amendment_a3_source',SOURCE_SHA)
    data=ep.runner.FilesystemDataSource(root).load_outer_train_only(config)
    train,v7=ep.inner_subject_partitions(data,n_splits=4)[0]
    data=ep.slice_outer_train(data,train)
    identities={'t24':list(train),'v7':list(v7),'outer':sorted(_e03_outer_subject_ids(root,config)),
                'source_train':[f'FD-R-{p}' for p in payload['provenance']['source_train']],
                'source_val':[f'FD-R-{p}' for p in payload['provenance']['source_validation']]}
    validate_identities(identities,source_target_evidence=SOURCE_TARGET_EVIDENCE)
    wrappers=[]
    for k,(fit,health) in enumerate(ep.inner_subject_partitions(data,n_splits=3)):
        policy=[{'train':list(t),'score':list(v)} for t,v in ep.inner_subject_partitions(ep.slice_outer_train(data,fit),n_splits=3)]
        wrappers.append({'id':k,'train':list(fit),'health':list(health),'policy_splits':policy})
    return {'id':IDENTITY,'identities':identities,'source_target_evidence':SOURCE_TARGET_EVIDENCE,
            'source_sha256':SOURCE_SHA,'wrappers':wrappers,'config':asdict(config),'contract':asdict(T1Config()),
            'subject_by_session':data.subject_by_session,'session_bounds_by_sid':data.session_bounds_by_sid,
            'development_only':True,'outer_calls':0}


def run_manifest(root):
    root=Path(root); output=root/OUTPUT
    if output.exists(): raise RuntimeError('attempt already exists; no overwrite')
    require_clean(root)
    snapshot=ep.protected_snapshot(root)
    try:
        manifest=build_manifest(root)
        files=subprocess.check_output(['git','ls-files','src','scripts'],cwd=root,text=True).splitlines()
        files=[p for p in files if p.endswith('.py')]+[SPEC,PLAN]
        write_json(output/'manifest.json',manifest)
        write_json(output/'registered_plan.json',{'id':IDENTITY,'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                   'code_files':{p:file_hash(root/p) for p in files},'manifest_sha256':file_hash(output/'manifest.json'),
                   'protected_digest':digest(snapshot),'outer_calls':0})
        if ep.protected_snapshot(root)!=snapshot: raise RuntimeError('protected artifacts changed')
        write_json(output/'state.json',{'status':'MANIFEST_READY','outer_calls':0})
    except BaseException as exc:
        write_json(output/'state.json',{'status':'INCOMPLETE','error':repr(exc),'outer_calls':0}); raise


def load_registration(root):
    output=Path(root)/OUTPUT
    plan=read_json(output/'registered_plan.json'); manifest=read_json(output/'manifest.json')
    verify_registered_files(root,plan)
    if (plan['id']!=IDENTITY or manifest['id']!=IDENTITY
            or file_hash(output/'manifest.json')!=plan['manifest_sha256']
            or digest(ep.protected_snapshot(root))!=plan['protected_digest']):
        raise RuntimeError('registration/protected artifacts changed')
    validate_identities(manifest['identities'],source_target_evidence=manifest['source_target_evidence'])
    return manifest


def run_audit(root):
    root=Path(root); manifest=load_registration(root); output=root/OUTPUT
    with phase(output,'MANIFEST_READY','AUDIT_PASSED'):
        audit=audit_data(root,manifest)
        if shutil.disk_usage(root).free<20*1024**3: raise RuntimeError('disk reserve insufficient')
        audit['cuda_available']=torch.cuda.is_available()
        audit['source_sha256']=manifest['source_sha256']
        audit['identity_sha256']=digest(manifest['identities'])
        write_json(output/'audit.json',audit)


def freeze_policies(output):
    output=Path(output); files={}
    if (output/'policy_phase_frozen.json').exists(): raise RuntimeError('policies already frozen')
    for k in range(3):
        path=output/f'wrapper{k}_policies.json'
        if not path.is_file(): raise RuntimeError('all policies required before health')
        row=read_json(path)
        if row['wrapper']!=k or set(row['policies'])!=set(ARMS): raise RuntimeError('policy contract differs')
        files[path.name]=file_hash(path)
    write_json(output/'policy_phase_frozen.json',{'files':files,'health_files_read':0,'outer_calls':0})


def read_frozen_policies(output):
    output=Path(output); marker=output/'policy_phase_frozen.json'
    if not marker.is_file(): raise RuntimeError('policies not frozen')
    record=read_json(marker)
    expected={f'wrapper{k}_policies.json' for k in range(3)}
    if set(record['files'])!=expected or record['health_files_read']!=0:
        raise RuntimeError('policy freeze contract changed')
    result={}
    for k in range(3):
        name=f'wrapper{k}_policies.json'
        if file_hash(output/name)!=record['files'][name]: raise RuntimeError('frozen policies changed')
        row=read_json(output/name)
        if row['wrapper']!=k or set(row['policies'])!=set(ARMS): raise RuntimeError('frozen policy identity changed')
        result[k]=row['policies']
    return result


class Resources:
    def __init__(self,root,output,*,new=False):
        self.root=Path(root); self.output=Path(output)
        path=self.output/'resources.json'
        self.row=({'started_epoch':time.time(),'producer_fits':0,'base_fits':0,'neural_fits':0,'steps':0,
                   'neural_seconds':0.,'encoding_seconds':0.,'cpu_seconds':0.,'fits':[]}
                  if new else read_json(path))
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.free_cuda=torch.cuda.mem_get_info()[0] if self.device=='cuda' else None
        if self.device=='cuda': torch.cuda.reset_peak_memory_stats()
        self.check()

    def check(self):
        size=sum(p.stat().st_size for p in self.output.rglob('*') if p.is_file() and 'tokens' not in p.parts)
        token_size=sum(p.stat().st_size for p in (self.output/'tokens').glob('*') if p.is_file())
        if (time.time()-self.row['started_epoch']>36*3600 or self.row['neural_seconds']>12*3600
                or self.row['encoding_seconds']>2*3600 or self.row['cpu_seconds']>12*3600
                or size>256*1024**2 or token_size>2*1024**3 or shutil.disk_usage(self.root).free<20*1024**3):
            raise RuntimeError('registered resource budget exceeded')
        if self.free_cuda is not None and torch.cuda.max_memory_allocated()>.8*self.free_cuda:
            raise RuntimeError('CUDA reserve exceeded')

    def save(self):
        self.row['device']=self.device
        self.row['peak_cuda_bytes']=torch.cuda.max_memory_allocated() if self.device=='cuda' else None
        write_json(self.output/'resources.json',self.row)


def validate_runtime_config(config,registered):
    if digest(asdict(config))!=digest(registered): raise RuntimeError('runtime config changed')


def load_runtime(root,manifest):
    config=ep.registered_e03_config(0,workers=4)
    validate_runtime_config(config,manifest['config'])
    data=ep.runner.FilesystemDataSource(root).load_outer_train_only(config)
    data=ep.slice_outer_train(data,manifest['identities']['t24'])
    if data.subject_by_session!=manifest['subject_by_session']: raise RuntimeError('session ownership changed')
    payload=ep.load_source_artifact(Path(root)/'outputs/challenger_70/representation_v2/amendment_a3_source',SOURCE_SHA)
    encoder=ep.MotionBranch().token_encoder
    encoder.load_state_dict(payload['encoder'],strict=True); encoder.requires_grad_(False); encoder.eval()
    return data,config,encoder


def sequences_for(root,ledger,manifest,encoder,resources):
    audit={row['sid']:row for row in read_json(resources.output/'audit.json')['sessions']}
    allowed_subjects=set(ledger.score_subjects)
    allowed_sids={s for s,owner in manifest['subject_by_session'].items() if owner in allowed_subjects}
    sessions={}; sequences=[]; started=time.perf_counter()
    for candidate,subject in zip(ledger.candidates,ledger.groups):
        event=candidate.event; sid=event.sid
        if subject not in allowed_subjects or sid not in allowed_sids: raise ValueError('sequence subject not allowed')
        if sid not in sessions:
            bounds=manifest['session_bounds_by_sid'][sid]
            meta={'schema':'a6-token-v1','raw_sha256':audit[sid]['sha256'],
                  'encoder_sha256':encoder_hash(encoder),'normalization':'unitfree',
                  'fit_subjects':manifest['identities']['source_train'],'source_checkpoint':SOURCE_SHA,
                  'start_ms':int(bounds[0]),'end_ms':int(bounds[1]),'sid':sid,
                  'preprocessing':{'sampling_hz':25,'token_ms':5000,'max_gap_ms':200,'min_coverage':.8,'gravity_align':False}}
            sessions[sid]=load_or_encode(root=root,sid=sid,allowed_sids=allowed_sids,encoder=encoder,
                    device=resources.device,cache_root=resources.output/'tokens',metadata=meta)
        tokens=sessions[sid]; chunks=[]
        for chunk in candidate_chunks(tokens,event.start_ms,event.end_ms):
            idx=chunk.token_indices
            chunks.append({'region':chunk.region,'indices':idx,'embeddings':tokens.embeddings[idx],
                           'coverage':tokens.coverage[idx],'times':tokens.center_ms[idx],'valid':tokens.valid[idx]})
        sequences.append({'sid':sid,'subject':str(subject),'start_ms':event.start_ms,'chunks':chunks})
    resources.row['encoding_seconds']+=time.perf_counter()-started
    resources.check(); resources.save()
    return sequences


def split_scores(root,data,config,train,score,excluded,manifest,encoder,resources,prefix,split_id):
    resources.check(); graph={}; started=time.perf_counter()
    t,v=ep.build_candidate_ledgers(data,config,train,score,fit_audit=graph)
    base=ep.verifier_probabilities(t.features,t.labels,v.features,seed=config.seed)
    resources.row['cpu_seconds']+=time.perf_counter()-started
    resources.row['producer_fits']+=10; resources.row['base_fits']+=2
    train_sequences=sequences_for(root,t,manifest,encoder,resources)
    score_sequences=sequences_for(root,v,manifest,encoder,resources)
    valid=np.asarray([any(c['valid'].any() for c in s['chunks']) for s in score_sequences])
    graph['source']={'fit_subjects':manifest['identities']['source_train'],'parents':[]}
    graph['base']={'fit_subjects':list(train),'parents':['candidates']}
    scores={'b':base}; heads={}; components={}
    for arm in ('p','c','t'):
        print(f'A6 {prefix} arm={arm} train={len(t.labels)} score={len(v.labels)}',flush=True)
        fitted=fit_head(arm,train_sequences,t.labels,t.groups,seed=20261012+split_id,
                        device=resources.device,resource_callback=resources.check)
        motion=predict_head(fitted,score_sequences); components[arm]=motion.tolist()
        scores[arm]=fuse_scores(base,motion,valid)
        sampling=fitted['sampling_sha256']
        if heads and sampling!=next(iter(heads.values()))['sampling_sha256']:
            raise RuntimeError('paired sampling changed between arms')
        summary={k:value for k,value in fitted.items() if k not in {'model','center','scale'}}
        heads[arm]=summary
        torch.save({'state_dict':{k:value.cpu() for k,value in fitted['model'].state_dict().items()},
                    'center':torch.tensor(fitted['center']),'scale':torch.tensor(fitted['scale']),
                    'provenance':summary},resources.output/f'{prefix}_{arm}.pt')
        resources.row['neural_fits']+=1; resources.row['steps']+=fitted['steps']
        resources.row['neural_seconds']+=fitted['elapsed_seconds']
        resources.row['fits'].append({'split':prefix,'arm':arm,**summary})
        graph[arm]={'fit_subjects':list(train),'parents':['source','candidates']}
        resources.check(); resources.save(); del fitted; gc.collect()
    graph['score']={'fit_subjects':[],'parents':['base','p','c','t']}
    ep.check_graph(graph,'score',set(score)|set(excluded)|set(manifest['identities']['source_val']),
                   set(train)|set(manifest['identities']['source_train']))
    evidence={'train':list(train),'score':list(score),'excluded':sorted(excluded),'graph':graph,
              'candidates':[asdict(c) for c in v.candidates],'groups':v.groups.tolist(),'labels':v.labels.tolist(),
              'scores':{a:p.tolist() for a,p in scores.items()},'components':components,
              'motion_valid':valid.tolist(),'heads':heads,'outer_calls':0}
    write_json(resources.output/f'{prefix}_scores.json',evidence)
    return v,scores


def run_policy(root):
    root=Path(root); require_clean(root); manifest=load_registration(root); output=root/OUTPUT
    require_committed(root,output/'manifest.json'); require_committed(root,output/'audit.json')
    with phase(output,'AUDIT_PASSED','POLICIES_FROZEN'):
        resources=Resources(root,output,new=True); data,config,encoder=load_runtime(root,manifest)
        resources.save()
        protected=set(manifest['identities']['v7'])|set(manifest['identities']['outer'])
        with threadpool_limits(limits=4):
            for wrapper in manifest['wrappers']:
                k=wrapper['id']; candidates=[]; groups=[]; predictions={a:[] for a in ARMS}
                train_data=ep.slice_outer_train(data,wrapper['train'])
                for j,part in enumerate(wrapper['policy_splits']):
                    v,scores=split_scores(root,train_data,config,part['train'],part['score'],protected|set(wrapper['health']),
                            manifest,encoder,resources,f'wrapper{k}_policy{j}',k*4+j)
                    candidates.extend(v.candidates); groups.extend(v.groups)
                    for a in ARMS: predictions[a].extend(scores[a])
                if set(groups)!=set(wrapper['train']): raise RuntimeError('policy OOF coverage differs')
                policies={a:ep.choose_policy(candidates,np.asarray(predictions[a]),groups,train_data.train_truths,config) for a in ARMS}
                write_json(output/f'wrapper{k}_policies.json',{'wrapper':k,'policies':policies,'outer_calls':0})
        freeze_policies(output)
        resources.save()


def run_health(root):
    root=Path(root); output=root/OUTPUT
    # Check the gate before loading registration, raw data or learned ancestors.
    policies=read_frozen_policies(output)
    require_clean(root); require_committed(root,output/'policy_phase_frozen.json')
    manifest=load_registration(root)
    with phase(output,'POLICIES_FROZEN','HEALTH_COMPLETE'):
        from .reporting import make_record
        resources=Resources(root,output); data,config,encoder=load_runtime(root,manifest)
        excluded=set(manifest['identities']['v7'])|set(manifest['identities']['outer'])
        records=[]
        with threadpool_limits(limits=4):
            for wrapper in manifest['wrappers']:
                k=wrapper['id']; read_frozen_policies(output)
                v,scores=split_scores(root,data,config,wrapper['train'],wrapper['health'],excluded,
                        manifest,encoder,resources,f'wrapper{k}_health',k*4+3)
                predictions={a:ep.decode_policy(v.candidates,scores[a],v.groups,policies[k][a]) for a in ARMS}
                health=ep.slice_outer_train(data,wrapper['health'])
                record=make_record(k,predictions,health.train_truths,health.subject_by_session,v.candidates,v.labels,scores)
                records.append(record); write_json(output/f'wrapper{k}_health.json',record)
        validate_fit_counts({k:resources.row[k] for k in ('producer_fits','base_fits','neural_fits','steps')})
        resources.check(); resources.save()
        write_json(output/'records.json',records)


def run_report(root):
    root=Path(root); output=root/OUTPUT; load_registration(root)
    with phase(output,'HEALTH_COMPLETE','COMPLETE'):
        from .reporting import summarize,write_report
        summary=summarize(read_json(output/'records.json'))
        summary['resources']=read_json(output/'resources.json')
        write_json(output/'summary.json',summary); write_report(output,summary)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--stage',choices=('manifest','audit','policy','health','report'),required=True)
    args=parser.parse_args()
    globals()['run_'+args.stage](args.root.resolve())


if __name__=='__main__': main()
