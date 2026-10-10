"""S10 train-only controls; S11 runs one preregistered, train-selected recipe."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch
from torch import nn

from scripts.research.representation_v2 import adapt_probe as ap
from scripts.research.representation_v2 import optimization_audit as audit

ep=ap.ep
OPTIONS={'unclipped':{'batch_size':4,'clip_norm':None},'larger_batch':{'batch_size':32,'clip_norm':1.}}
PILOT='outputs/challenger_70/representation_v2/amendment_a3_optimizer_controls'


def select_optimizer_arm(initial,arms,initial_source):
    if set(arms)!=set(OPTIONS):raise ValueError('exactly two registered controls required')
    subjects=set(initial['subject_bce']);eligible={};diagnostics={}
    if len(subjects)!=11:raise ValueError('registered eleven-subject training set required')
    for name in OPTIONS:
        row=arms[name];m=row['metrics']
        values=[initial['subject_weighted_bce'],initial_source,m['subject_weighted_bce'],row['source_bce'],
                *initial['subject_bce'].values(),*m['subject_bce'].values()]
        if set(m['subject_bce'])!=subjects or not np.isfinite(values).all():raise ValueError('aligned finite train metrics required')
        improvement=initial['subject_weighted_bce']-m['subject_weighted_bce']
        nonworse=sum(m['subject_bce'][s]<=initial['subject_bce'][s] for s in subjects)
        source_rise=row['source_bce']-initial_source
        eligible[name]=bool(improvement>=.01-1e-12 and nonworse>=6 and source_rise<=.03+1e-12)
        diagnostics[name]={'bce_improvement':improvement,'nonworse_subjects':nonworse,'source_bce_rise':source_rise}
    return {'chosen':next((name for name in OPTIONS if eligible[name]),None),'eligible':eligible,'diagnostics':diagnostics}


def source_train_bce(encoder,head,bank,device):
    logits=[];encoder.eval();head.to(device).eval()
    with torch.inference_mode():
        for left in range(0,len(bank['labels']),512):
            x=torch.as_tensor(bank['values'][left:left+512],device=device)
            z=encoder(x,torch.ones(len(x),125,dtype=torch.bool,device=device))
            logits.extend(head(z).squeeze(-1).cpu().numpy())
    loss=np.logaddexp(0,np.asarray(logits))-bank['labels']*np.asarray(logits)
    return float(np.mean([np.mean([loss[(bank['groups']==s)&(bank['labels']==c)].mean()
        for c in np.unique(bank['labels'][bank['groups']==s])]) for s in np.unique(bank['groups'])]))


def pilot(root):
    output=root/PILOT
    if output.exists():raise RuntimeError('S10 attempt exists; no silent overwrite')
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():raise RuntimeError('commit controls first')
    started=time.perf_counter();snapshot=ep.protected_snapshot(root)
    previous=root/'outputs/challenger_70/representation_v2/amendment_a3_adapt'
    metadata=json.loads((previous/'fold0_policy0_scores.json').read_text())
    saved=json.loads((previous/'fold0_policy0_adaptation.json').read_text())
    for name in ('fold0_policy0_scores.json','fold0_policy0_adaptation.json'):
        raw=subprocess.check_output(['git','show','8206cc6:outputs/challenger_70/representation_v2/amendment_a3_adapt/'+name],cwd=root)
        if (previous/name).read_bytes()!=raw:raise RuntimeError('sealed S8 input mismatch')
    train=metadata['train'];device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4)
    free=None
    if device=='cuda':
        free,_=torch.cuda.mem_get_info();torch.cuda.reset_peak_memory_stats()
        if free<1280*1024**2:raise RuntimeError('insufficient initial CUDA headroom')
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    spec=root/'docs/challenge_070_amendment_rv2_a3_optimizer_controls.md'
    plan={'id':'RV2-A3-S10-001','code_sha':code,'train_subjects':train,'options':OPTIONS,'seed':20261011,
        'steps_per_arm':1000,'source_weight':.25,'producer_fits':8,'readout_fits':2,'event_verifier_fits':0,
        'encoder_optimizer_steps':2000,'health_files_read':0,'outer_calls':0,'device':device,
        'source_checkpoint_sha256':ep.REGISTERED_SOURCE_SHA,'spec_sha256':hashlib.sha256(spec.read_bytes()).hexdigest()}
    ep.write_json(output/'plan.json',plan)
    try:
        payload=ep.load_source_artifact(root/'outputs/challenger_70/representation_v2/amendment_a3_source',ep.REGISTERED_SOURCE_SHA)
        config=ep.registered_e03_config(0,workers=4)
        data=ep.slice_outer_train(ep.runner.FilesystemDataSource(root).load_outer_train_only(config),train);graph={}
        ledger,validation=ep.build_candidate_ledgers(data,config,train,(),fit_audit=graph)
        assert validation is None
        ep.check_graph(graph,'candidates',set(metadata['score'])|set(metadata['excluded']),set(train))
        bags=ep.unitfree_bags(ep.load_candidate_bags(root,ledger.candidates,config=ep.PreprocessConfig(gravity_align=False)))
        valid=np.array([b.token_mask.any() for b in bags]);y=ledger.labels[valid];groups=ledger.groups[valid]
        if set(map(str,groups))!=set(saved['fit_subjects']):raise RuntimeError('training owners differ')
        initial=ep.MotionBranch().token_encoder;initial.load_state_dict(payload['encoder']);initial.requires_grad_(False).to(device).eval()
        x=np.array([ap.pool_ordered(z,b) for z,b in zip(ap.encode_tokens(initial,bags,device),bags)])[valid]
        readout=saved['initial_readout'];initial_metrics=audit.fixed_metrics(audit.saved_logits(x,readout),y,groups)
        s9=json.loads((root/'outputs/challenger_70/representation_v2/amendment_a3_optimization/train_metrics.json').read_text())
        np.testing.assert_allclose(initial_metrics['subject_weighted_bce'],s9['metrics']['initial_encoder_initial_head']['subject_weighted_bce'],rtol=0,atol=1e-10)
        bank,source_head=ap.source_training_bank(root,payload,output)
        initial_source=source_train_bce(initial,source_head,bank,device)
        rows={};model_hashes={}
        for name,options in OPTIONS.items():
            print('S10 train-only '+name,flush=True)
            model,report=ap.adapt_encoder(initial,source_head,bags,ledger.labels,ledger.groups,bank,
                train_subjects=set(train),seed=20261011,device=device,steps=1000,**options)
            for key in ('mean','scale','coefficient','intercept'):
                np.testing.assert_allclose(report['initial_readout'][key],readout[key],rtol=1e-7,atol=1e-8)
            features=np.array([ap.pool_ordered(z,b) for z,b in zip(ap.encode_tokens(model,bags,device),bags)])[valid]
            final_readout={**readout,'coefficient':report['final_target_head']['weight'],'intercept':report['final_target_head']['bias']}
            rows[name]={'metrics':audit.fixed_metrics(audit.saved_logits(features,final_readout),y,groups),
                        'source_bce':source_train_bce(model,source_head,bank,device),'wall_seconds':report['wall_seconds'],
                        'sampled_target_candidate_examples':report['sampled_target_candidate_examples']}
            path=output/(name+'_adaptation.json');ep.write_json(path,report)
            model_hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            print(name+' fixed train '+json.dumps(rows[name]),flush=True)
        selection=select_optimizer_arm(initial_metrics,rows,initial_source)
        peak=torch.cuda.max_memory_allocated() if device=='cuda' else None
        size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        if time.perf_counter()-started>3600 or size>16*1024**2 or shutil.disk_usage(root).free<20*1024**3 or (peak is not None and peak>.8*free):
            raise RuntimeError('pilot resource budget exceeded')
        if ep.protected_snapshot(root)!=snapshot:raise RuntimeError('protected release changed')
        ep.write_json(output/'train_comparison.json',{'initial':initial_metrics,'initial_source_bce':initial_source,
            'arms':rows,'producer_graph':graph,'valid_candidates':int(valid.sum()),'invalid_candidates':int((~valid).sum())})
        selection.update({'id':'RV2-A3-S10-001','plan_sha256':hashlib.sha256((output/'plan.json').read_bytes()).hexdigest(),
                          'comparison_sha256':hashlib.sha256((output/'train_comparison.json').read_bytes()).hexdigest(),
                          'model_file_sha256':model_hashes,'health_files_read':0,'outer_calls':0})
        ep.write_json(output/'selection.json',selection)
        ep.write_json(output/'state.json',{'status':'S10_OPTIMIZER_SELECTED' if selection['chosen'] else 'S10_NO_SUPPORTED_OPTIMIZER_REPAIR',
            'chosen':selection['chosen'],'wall_seconds':time.perf_counter()-started,'peak_cuda_allocated_bytes':peak,
            'protected_unchanged':True,'resource_ok':True,'health_files_read':0,'outer_calls':0,'neural_optimizer_steps':2000,
            'persistent_token_cache_bytes':0})
        print(json.dumps(selection),flush=True)
    except Exception as exc:
        ep.write_json(output/'state.json',{'status':'S10_INCOMPLETE','error':str(exc),'outer_calls':0})
        raise


def load_pilot_selection(output):
    selection=json.loads((output/'selection.json').read_text());plan=json.loads((output/'plan.json').read_text())
    state=json.loads((output/'state.json').read_text())
    if (selection['id']!='RV2-A3-S10-001' or selection['health_files_read']!=0 or selection['outer_calls']!=0 or
        plan['id']!='RV2-A3-S10-001' or plan['options']!=OPTIONS or plan['steps_per_arm']!=1000 or
        plan['source_weight']!=.25 or plan['seed']!=20261011 or plan['source_checkpoint_sha256']!=ep.REGISTERED_SOURCE_SHA):
        raise RuntimeError('unregistered pilot selection')
    expected_status='S10_OPTIMIZER_SELECTED' if selection['chosen'] else 'S10_NO_SUPPORTED_OPTIMIZER_REPAIR'
    if (state['status']!=expected_status or state['chosen']!=selection['chosen'] or
        state.get('protected_unchanged') is not True or state.get('resource_ok') is not True or
        state['health_files_read']!=0 or state['outer_calls']!=0):raise RuntimeError('pilot incomplete or failed checks')
    if hashlib.sha256((output/'plan.json').read_bytes()).hexdigest()!=selection['plan_sha256']:
        raise RuntimeError('pilot plan changed')
    if hashlib.sha256((output/'train_comparison.json').read_bytes()).hexdigest()!=selection['comparison_sha256']:
        raise RuntimeError('pilot comparison changed')
    if set(selection['model_file_sha256'])!={n+'_adaptation.json' for n in OPTIONS}:raise RuntimeError('pilot models incomplete')
    for name,digest in selection['model_file_sha256'].items():
        if name not in {n+'_adaptation.json' for n in OPTIONS} or hashlib.sha256((output/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('pilot model evidence changed')
    comparison=json.loads((output/'train_comparison.json').read_text())
    if set(comparison['initial']['subject_bce'])!=set(plan['train_subjects']):raise RuntimeError('pilot owners changed')
    expected=select_optimizer_arm(comparison['initial'],comparison['arms'],comparison['initial_source_bce'])
    for key in ('chosen','eligible','diagnostics'):
        if selection[key]!=expected[key]:raise RuntimeError('pilot choice is not registered rule')
    return selection


def event(root):
    output=root/PILOT;selection=load_pilot_selection(output)
    if selection['chosen'] is None:
        print('S11 skipped: both train-only controls failed',flush=True);return
    options=OPTIONS[selection['chosen']]
    def prepare(root,payload,path,device):
        ep.write_json(path/'pilot_selection_identity.json',{'chosen':selection['chosen'],'options':options,
            'selection_sha256':hashlib.sha256((output/'selection.json').read_bytes()).hexdigest(),
            'pilot_plan_sha256':selection['plan_sha256']})
        return ap.prepare(root,payload,path,device,adaptation_options=options)
    ap.jp.run(root,experiment={'id':'RV2-A3-S11-001','label':'S11','output':'amendment_a3_optimization_event',
        'spec':'challenge_070_amendment_rv2_a3_optimizer_controls.md','prepare':prepare,'neural_optimizer_steps':12000,
        'adaptation_readout_fits':12,'max_wall_seconds':21600,'max_work_seconds':10800,
        'elapsed_field':'adaptation_and_encoding_seconds','pass_status':'S11_SUPPORTS_WIDER_INNER_EVENT_VALIDATION',
        'fail_status':'S11_NO_SUPPORTED_OPTIMIZATION_EVENT_GAIN'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('pilot','event'),required=True)
    args=parser.parse_args()
    with ep.threadpool_limits(limits=4):
        (pilot if args.stage=='pilot' else event)(Path(__file__).resolve().parents[3])
