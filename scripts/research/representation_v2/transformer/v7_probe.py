"""Frozen T24 refit -> one V7 evaluation, without an outer/release entrypoint."""
import argparse
from dataclasses import asdict
import gc
from pathlib import Path
import subprocess
import time

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .. import event_probe as ep
from ..crossfit import build_candidate_ledgers
from ..stability_audit import restore_candidate
from ..validation_metrics import matched_indices
from .cache import file_hash
from .contract import SOURCE_SHA, digest
from .heads import build_head
from .probe import (Resources, load_registration, phase, read_json, require_clean,
                    require_committed, sequences_for, verify_registered_files, write_json)
from .reporting import metrics
from .training import fit_head, predict_head, fuse_scores
from .v7_contract import ARMS, ID, OUTPUT, primary_gate, seal, validate_oof, verify_seal
from .v7_models import SubsetSource, base_probability, canonical_session_ids, fit_producers_and_base, score_ledger

T1='outputs/challenger_70/representation_v2/amendment_a6_transformer/t1'
SPEC='docs/challenge_070_amendment_a6_v7_validation.md'
FROZEN_FILES=('manifest.json','registry.json','policies.json','models.joblib',
              'c.pt','t.pt','train_provenance.json')


class V7Resources(Resources):
    def check(self,*,active_neural_seconds=0.,active_encoding_seconds=0.):
        super().check(active_neural_seconds=active_neural_seconds,active_encoding_seconds=active_encoding_seconds)
        if (time.time()-self.row['started_epoch']>4*3600
                or self.row['neural_seconds']+active_neural_seconds>2*3600
                or self.row['encoding_seconds']+active_encoding_seconds>3600
                or self.row['cpu_seconds']>3600):
            raise RuntimeError('V7 resource budget exceeded')


def _registered(root):
    output=Path(root)/OUTPUT
    registry=read_json(output/'registry.json')
    verify_registered_files(root,registry)
    if (registry['id']!=ID or digest(ep.protected_snapshot(root))!=registry['protected_digest']
            or file_hash(output/'manifest.json')!=registry['manifest_sha256']):
        raise ValueError('V7 registration/release changed')
    return read_json(output/'manifest.json')


def _runtime(root,manifest,subjects):
    allowed=set(subjects)
    sids={s for s,p in manifest['subject_by_session'].items() if p in allowed}
    config=ep.registered_e03_config(0,workers=4)
    data=SubsetSource(root,sids).load_outer_train_only(config)
    if set(data.subject_by_session.values())!=allowed: raise ValueError('runtime subject coverage differs')
    if any(manifest['subject_by_session'].get(s)!=p for s,p in data.subject_by_session.items()):
        raise ValueError('runtime session ownership differs')
    for sid,bounds in data.session_bounds_by_sid.items():
        if list(bounds)!=manifest['session_bounds_by_sid'][sid]: raise ValueError('session bounds changed')
    if digest(asdict(config))!=digest(manifest['config']): raise ValueError('runtime config differs')
    payload=ep.load_source_artifact(Path(root)/'outputs/challenger_70/representation_v2/amendment_a3_source',SOURCE_SHA)
    encoder=ep.MotionBranch().token_encoder
    encoder.load_state_dict(payload['encoder'],strict=True); encoder.requires_grad_(False); encoder.eval()
    return data,config,encoder


def run_register(root):
    root=Path(root); output=root/OUTPUT
    if output.exists(): raise RuntimeError('V7 attempt already exists')
    require_clean(root)
    original=load_registration(root)
    ids=original['identities']
    from src.data import manifests
    index=manifests.load_sensor_index()
    subjects=set(ids['t24'])|set(ids['v7'])
    canonical=canonical_session_ids(root)
    rows=[r for _,r in index.iterrows() if str(r['externalid']) in subjects and str(r['session_id']) in canonical]
    if {str(r['session_id']) for r in rows}!=canonical:
        raise ValueError('canonical train coverage contains unknown or unauthorized sessions')
    manifest={**original,'id':ID,'primary':'c','secondary':'t',
        'subject_by_session':{str(r['session_id']):str(r['externalid']) for r in rows},
        'session_bounds_by_sid':{str(r['session_id']):[int(r['timeStamp.startTime']),int(r['timeStamp.endTime'])] for r in rows},
        'expected_fits':{'producer_fits':10,'base_fits':2,'neural_fits':2,'steps':2000},
        'historical_exposure':'same competition cohort; prior eligible-truth loader could read V7 timestamps/masks; no pristine-test claim',
        'refit_seed':20261024,'outer_calls':0,'dist_changed':False}
    oof_names=[f'{T1}/wrapper{k}_health_scores.json' for k in range(3)]
    for name in oof_names: require_committed(root,root/name)
    validate_oof([read_json(root/n) for n in oof_names],set(ids['t24']),set(ids['source_train']),
                 set(ids['v7'])|set(ids['outer'])|set(ids['source_val']))
    tracked=subprocess.check_output(['git','ls-files','scripts','src'],cwd=root,text=True).splitlines()
    features=[f'cache/{folder}/fold0_{role}.npz' for folder in ('slide','micro15')
              for role in ('train','meal_train','no_meal_train')]
    names=[n for n in tracked if n.endswith('.py')]+[SPEC]+oof_names+[f'{T1}/manifest.json',f'{T1}/registered_plan.json']+features
    output.mkdir(parents=True)
    try:
        write_json(output/'manifest.json',manifest)
        write_json(output/'registry.json',{'id':ID,'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            'code_files':{n:file_hash(root/n) for n in names},'manifest_sha256':file_hash(output/'manifest.json'),
            'protected_digest':digest(ep.protected_snapshot(root))})
        audit=[]
        for sid,subject in sorted(manifest['subject_by_session'].items()):
            stat=(root/'cache/sessions'/f'{sid}.npz').stat()
            audit.append({'sid':sid,'subject':subject,'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns})
        write_json(output/'audit.json',{'sessions':audit,'raw_bytes_read':False,'raw_arrays_decoded':False})
        write_json(output/'state.json',{'status':'REGISTERED','outer_calls':0})
    except BaseException as exc:
        write_json(output/'state.json',{'status':'INCOMPLETE','error':repr(exc),'outer_calls':0}); raise


def run_train(root):
    root=Path(root); output=root/OUTPUT; require_clean(root); manifest=_registered(root)
    require_committed(root,output/'registry.json'); require_committed(root,output/'audit.json')
    with phase(output,'REGISTERED','FROZEN'):
        resources=V7Resources(root,output,new=True); resources.save()
        data,config,encoder=_runtime(root,manifest,manifest['identities']['t24'])
        rows=[read_json(root/T1/f'wrapper{k}_health_scores.json') for k in range(3)]
        ids=manifest['identities']
        validate_oof(rows,set(ids['t24']),set(ids['source_train']),set(ids['v7'])|set(ids['outer'])|set(ids['source_val']))
        candidates=[restore_candidate(c) for row in rows for c in row['candidates']]
        groups=[g for row in rows for g in row['groups']]
        with threadpool_limits(limits=4):
            policies={a:ep.choose_policy(candidates,np.array([p for row in rows for p in row['scores'][a]]),
                        groups,data.train_truths,config) for a in ARMS}
            write_json(output/'policies.json',{'policies':policies,'selection_population':'T24 reused subject-OOF','primary':'c'})
            started=time.perf_counter(); graph={}
            ledger,unused=build_candidate_ledgers(data,config,ids['t24'],(),fit_audit=graph)
            if unused is not None: raise RuntimeError('unexpected score ledger before freeze')
            models=fit_producers_and_base(data,config,ledger)
            resources.row.update(producer_fits=10,base_fits=2)
            resources.row['cpu_seconds']+=time.perf_counter()-started; resources.check(); resources.save()
            joblib.dump(models,output/'models.joblib')
            sequences=sequences_for(root,ledger,manifest,encoder,resources)
            graph['source']={'fit_subjects':ids['source_train'],'parents':[]}
            graph['base']={'fit_subjects':ids['t24'],'parents':['candidates']}
            histories={}
            for arm in ('c','t'):
                print(f'V7 final refit arm={arm} candidates={len(ledger.labels)}',flush=True)
                started=time.perf_counter()
                fitted=fit_head(arm,sequences,ledger.labels,ledger.groups,seed=manifest['refit_seed'],device=resources.device,
                    resource_callback=lambda:resources.check(active_neural_seconds=time.perf_counter()-started))
                info={k:v for k,v in fitted.items() if k not in {'model','center','scale'}}
                if histories and info['sampling_sha256']!=histories['c']['sampling_sha256']:
                    raise ValueError('paired refit sampling differs')
                histories[arm]=info
                torch.save({'state_dict':{k:v.cpu() for k,v in fitted['model'].state_dict().items()},
                            'center':torch.tensor(fitted['center']),'scale':torch.tensor(fitted['scale']),
                            'provenance':info},output/f'{arm}.pt')
                resources.row['neural_fits']+=1; resources.row['steps']+=fitted['steps']
                resources.row['neural_seconds']+=time.perf_counter()-started
                resources.row['fits'].append({'split':'T24_refit','arm':arm,**info})
                resources.check(); resources.save(); del fitted; gc.collect()
                graph[arm]={'fit_subjects':ids['t24'],'parents':['source','candidates']}
            graph['final']={'fit_subjects':ids['t24'],'parents':['base','c','t']}
            ep.check_graph(graph,'final',set(ids['v7'])|set(ids['outer'])|set(ids['source_val']),
                           set(ids['t24'])|set(ids['source_train']))
            if {k:resources.row[k] for k in manifest['expected_fits']}!=manifest['expected_fits']:
                raise ValueError('V7 actual fit budget differs')
            write_json(output/'train_provenance.json',{'graph':graph,'heads':histories,'v7_raw_opened':False,
                'candidate_count':len(ledger.labels),'micro_threshold':ledger.micro_threshold,'outer_calls':0})
            seal(output,list(FROZEN_FILES)); resources.save()


def summarize_v7(predictions,truths,mapping,expected,candidates):
    if len(expected)!=7 or len(set(expected))!=7 or set(mapping.values())!=set(expected):
        raise ValueError('seven unique V7 subjects required')
    if set(predictions)!=set(ARMS): raise ValueError('three registered arms required')
    if any(e.sid not in mapping for values in predictions.values() for e in values) or any(t.sid not in mapping for t in truths):
        raise ValueError('missing event ownership')
    events={a:metrics(predictions[a],truths) for a in ARMS}; subjects=[]; short={}; diagnostics={}
    for subject in sorted(expected):
        sids={s for s,p in mapping.items() if p==subject}; ts=[t for t in truths if t.sid in sids]
        subjects.append({'subject':subject,'events':{a:metrics([p for p in predictions[a] if p.sid in sids],ts) for a in ARMS}})
    for a in ARMS:
        _,matched=matched_indices(predictions[a],truths)
        ntrue=sum(t.end_ms-t.start_ms<600000 for t in truths)
        ntp=sum(truths[i].end_ms-truths[i].start_ms<600000 for i in matched)
        short[a]={'n_true':ntrue,'n_tp':ntp,'recall':ntp/ntrue if ntrue else None}
        gt=[s for s in subjects if s['events'][a]['n_true']>0]; f1=[s['events'][a]['f1'] for s in gt]
        delta=[s['events'][a]['f1']-s['events']['b']['f1'] for s in gt]
        diagnostics[a]={'median':float(np.median(f1)) if f1 else None,
            'iqr':np.percentile(f1,[25,75]).tolist() if f1 else None,'p10':float(np.percentile(f1,10)) if f1 else None,
            'improved':sum(d>0 for d in delta),'worsened':sum(d<0 for d in delta),'tied':sum(d==0 for d in delta),
            'no_gt_subjects':sum(s['events'][a]['n_true']==0 for s in subjects),
            'no_gt_fp':sum(s['events'][a]['fp'] for s in subjects if s['events'][a]['n_true']==0)}
    reachable=metrics(candidates,truths)
    return {'events':events,'subject_events':subjects,'subject_diagnostics':diagnostics,'short_events':short,
        'candidate_recall':{'n_true':reachable['n_true'],'n_reachable':reachable['n_tp'],'recall':reachable['sensitivity']},
        'gate':primary_gate(events),'development_validation':True,'pristine_test':False,'outer_calls':0,'dist_changed':False}


def run_evaluate(root):
    root=Path(root); output=root/OUTPUT
    frozen=verify_seal(output)  # Must precede registry/raw/model deserialization.
    if set(frozen['files'])!=set(FROZEN_FILES): raise ValueError('incomplete final-model freeze')
    require_clean(root); require_committed(root,output/'frozen.json'); manifest=_registered(root)
    with phase(output,'FROZEN','COMPLETE'):
        resources=V7Resources(root,output)
        data,config,encoder=_runtime(root,manifest,manifest['identities']['v7'])
        models=joblib.load(output/'models.joblib')  # Local trusted artifact, hashed before loading.
        if models['train']!=sorted(manifest['identities']['t24']): raise ValueError('model training identities differ')
        with threadpool_limits(limits=4):
            ledger=score_ledger(data,config,models)
            sequences=sequences_for(root,ledger,manifest,encoder,resources)
            valid=np.array([any(c['valid'].any() for c in s['chunks']) for s in sequences])
            scores={'b':base_probability(models,ledger.features)}
            for arm in ('c','t'):
                checkpoint=torch.load(output/f'{arm}.pt',map_location='cpu',weights_only=True)
                model=build_head(arm).to(resources.device); model.load_state_dict(checkpoint['state_dict'],strict=True)
                fitted={'model':model.eval(),'center':checkpoint['center'].numpy(),
                        'scale':checkpoint['scale'].numpy(),'device':resources.device}
                started=time.perf_counter()
                motion=predict_head(fitted,sequences,resource_callback=lambda:resources.check(active_neural_seconds=time.perf_counter()-started))
                scores[arm]=fuse_scores(scores['b'],motion,valid)
                resources.row['neural_seconds']+=time.perf_counter()-started
                del model,fitted; gc.collect()
            policies=read_json(output/'policies.json')['policies']
            predictions={a:ep.decode_policy(ledger.candidates,scores[a],ledger.groups,policies[a]) for a in ARMS}
            write_json(output/'v7_scores.json',{'scores':{a:p.tolist() for a,p in scores.items()},
                'candidates':[asdict(c) for c in ledger.candidates],'groups':ledger.groups.tolist(),
                'motion_valid':valid.tolist(),'predictions':{a:[asdict(p) for p in predictions[a]] for a in ARMS},
                'frozen_sha256':file_hash(output/'frozen.json'),'outer_calls':0})
            # Score/prediction evidence is persisted before GT metrics. Never tune here.
            summary=summarize_v7(predictions,data.train_truths,data.subject_by_session,
                                 manifest['identities']['v7'],[c.event for c in ledger.candidates])
            summary['score_sha256']=file_hash(output/'v7_scores.json')
            summary['truths']=[asdict(t) for t in data.train_truths]
            verify_seal(output); resources.check(); resources.save()
            if {k:resources.row[k] for k in manifest['expected_fits']}!=manifest['expected_fits']:
                raise ValueError('unexpected evaluation fitting')
            write_json(output/'summary.json',summary)
            lines=['# A6 frozen V7 validation','',
                'TCN primary; Transformer secondary. Same historical competition cohort, not pristine independent evidence. No outer/dist/release change.','',
                '| Arm | F1 | TP | FP | FN |','| --- | --- | --- | --- | --- |']
            for a in ARMS:
                m=summary['events'][a]; lines.append(f"| {a} | {m['f1']:.6f} | {m['n_tp']} | {m['fp']} | {m['fn']} |")
            lines+=['',f"Primary TCN delta: {summary['gate']['delta_f1']:+.6f}; gate passed: {summary['gate']['passed']}.",
                'Secondary success cannot override a failed primary gate. A pass only supports requesting separately approved outer evaluation.',
                '', 'Per-subject, short-event and candidate counts are in summary.json; final models/policies were committed before V7 inference.']
            (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--stage',choices=('register','train','evaluate'),required=True)
    args=parser.parse_args(); globals()['run_'+args.stage](args.root.resolve())


if __name__=='__main__': main()
