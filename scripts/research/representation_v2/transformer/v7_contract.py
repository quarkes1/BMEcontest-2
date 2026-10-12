"""A narrowly scoped, non-overwriting primary/secondary validation contract."""
from pathlib import Path

import numpy as np

from .cache import file_hash
from .probe import read_json, write_json
from ..event_probe import check_graph

OUTPUT='outputs/challenger_70/representation_v2/amendment_a6_transformer/v7'
ARMS=('b','c','t')
ID='RV2-A6-V7-001'


def seal(output, names):
    output=Path(output)
    if (output/'frozen.json').exists(): raise RuntimeError('freeze already exists')
    if not names or len(set(names))!=len(names): raise ValueError('invalid freeze files')
    for name in names:
        if Path(name).is_absolute() or '..' in Path(name).parts or not (output/name).is_file():
            raise ValueError('missing or invalid freeze file')
    write_json(output/'frozen.json',{'id':ID,'files':{n:file_hash(output/n) for n in names},
                                    'v7_evaluated':False,'outer_calls':0})


def verify_seal(output):
    output=Path(output)
    if not (output/'frozen.json').is_file(): raise RuntimeError('final models not frozen')
    row=read_json(output/'frozen.json')
    if row.get('id')!=ID or row.get('v7_evaluated') is not False or row.get('outer_calls')!=0:
        raise ValueError('invalid freeze identity')
    if not row.get('files'): raise ValueError('empty freeze')
    for name,sha in row['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('invalid freeze path')
        if not (output/name).is_file() or file_hash(output/name)!=sha:
            raise ValueError('frozen artifact changed: '+name)
    return row


def primary_gate(events):
    if set(events)!=set(ARMS): raise ValueError('three registered arms required')
    values=np.array([events[a]['f1'] for a in ARMS])
    if not np.isfinite(values).all() or ((values<0)|(values>1)).any(): raise ValueError('invalid F1')
    delta=float(values[1]-values[0])
    return {'primary':'c','passed':bool(delta>=.01-1e-12),'delta_f1':delta,
            'secondary_delta':float(values[2]-values[0]),'release_authorized':False}


def validate_oof(rows, t24, source, excluded, *, require_three=True):
    if require_three and len(rows)!=3: raise ValueError('three T24 score partitions required')
    seen=set()
    for row in rows:
        train=set(row['train']); score=set(row['score'])
        if not score or score&seen or train&score or train|score!=set(t24):
            raise ValueError('OOF subject coverage/overlap differs')
        if row.get('outer_calls')!=0 or not set(row['groups'])<=score:
            raise ValueError('invalid OOF ownership')
        if set(row['groups'])!=score: raise ValueError('OOF candidate subject missing')
        check_graph(row['graph'],'score',score|set(excluded),train|set(source))
        n=len(row['candidates']); valid=np.asarray(row['motion_valid'],bool)
        if valid.shape!=(n,) or len(row['labels'])!=n or len(row['groups'])!=n:
            raise ValueError('OOF row alignment differs')
        scores={a:np.asarray(row['scores'][a],float) for a in ARMS}
        for a,p in scores.items():
            if p.shape!=(n,) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
                raise ValueError('invalid OOF probability')
            if a!='b':
                motion=np.asarray(row['components'][a],float)
                if motion.shape!=(n,) or not np.isfinite(motion).all() or ((motion<0)|(motion>1)).any():
                    raise ValueError('invalid OOF motion probability')
                expected=np.where(valid,.75*scores['b']+.25*motion,scores['b'])
                if not np.allclose(p,expected,atol=1e-12,rtol=0): raise ValueError('OOF fusion changed')
        seen.update(score)
    if seen!=set(t24): raise ValueError('OOF subject coverage incomplete')
