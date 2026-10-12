"""Bounded full-session inference with span-faithful sampling and sealed caches."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.signal import butter, sosfiltfilt
import torch

from scripts.research.representation_v2.dataset import prepare_session
from scripts.research.representation_v2.unitfree import normalize_triads
from .contract import cache_key, T1Config
from .timeline import SessionTokens


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''): h.update(block)
    return h.hexdigest()


def encoder_hash(encoder):
    h=hashlib.sha256()
    for name,value in sorted(encoder.state_dict().items()):
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _require_sid(sid,allowed_sids):
    if sid not in allowed_sids: raise ValueError('session not allowed')
    if not isinstance(sid,str) or not sid or '/' in sid or '\\' in sid or sid in {'.','..'}:
        raise ValueError('invalid session identifier')


def encode_session(root,sid,allowed_sids,encoder,device,*,resource_callback=None):
    _require_sid(sid,allowed_sids)
    if resource_callback: resource_callback()
    if encoder.training or any(p.requires_grad for p in encoder.parameters()):
        raise ValueError('encoder must be frozen and in eval mode')
    before=encoder_hash(encoder)
    with np.load(Path(root)/'cache/sessions'/f'{sid}.npz',allow_pickle=False) as source:
        raw=SimpleNamespace(**{k:source[k] for k in ('acc','gyro','t_acc','imu_valid','row_rate')})
    prepared=prepare_session(raw)
    start=int(np.floor(prepared.first_ms))
    end=int(np.ceil(prepared.last_ms+1000/prepared.row_rate))
    starts=np.arange(start,end,5000,dtype=np.int64)
    n=len(starts); z=np.zeros((n,16),np.float32); coverage=np.zeros(n,np.float32)
    valid=np.zeros(n,bool); span_ids=np.full(n,-1,np.int64)
    # Filter on complete acquisition spans once. Compute chunk edges are not signal edges.
    filtered=[]
    if prepared.row_rate>20:
        sos=butter(4,10.,btype='lowpass',fs=prepared.row_rate,output='sos')
        for t,values in prepared.spans:
            filtered.append((t,sosfiltfilt(sos,values,axis=1) if len(t)>=32 else None))
    encoder=encoder.to(device)
    with torch.inference_mode():
        for left in range(0,n,360):
            if resource_callback: resource_callback()
            right=min(n,left+360); count=right-left
            grid=starts[left:right,None]+np.arange(125,dtype=np.int64)[None,:]*40
            values=np.zeros((count,6,125),np.float32); mask=np.zeros((count,125),bool)
            best=np.zeros(count,np.int64); owners=np.full(count,-1,np.int64)
            for span_id,(t,x) in enumerate(filtered):
                if x is None or t[-1]<grid[0,0] or t[0]>grid[-1,-1]: continue
                observed=(grid>=t[0])&(grid<=t[-1])&(grid<end)
                sizes=observed.sum(1); keep=np.flatnonzero(sizes>best)
                if not len(keep): continue
                # Strict > implements the deterministic earlier-span tie policy.
                for channel in range(6):
                    sampled=np.interp(grid[keep].ravel(),t,x[channel]).reshape(len(keep),125)
                    values[keep,channel]=np.where(observed[keep],sampled,0.)
                mask[keep]=observed[keep]; best[keep]=sizes[keep]; owners[keep]=span_id
            values,normal_valid=normalize_triads(values,mask)
            cov=mask.mean(1).astype(np.float32); keep=(cov>=.8)&normal_valid
            coverage[left:right]=cov; valid[left:right]=keep; span_ids[left:right]=owners
            indices=np.flatnonzero(keep)
            if len(indices):
                vectors=encoder(torch.as_tensor(values[indices],device=device),
                                torch.as_tensor(mask[indices],device=device)).cpu().numpy()
                if vectors.shape!=(len(indices),16) or not np.isfinite(vectors).all():
                    raise ValueError('invalid encoder output')
                z[left+indices]=vectors
    if encoder_hash(encoder)!=before: raise RuntimeError('frozen encoder changed')
    return SessionTokens(sid,starts+2500,span_ids,coverage,valid,z,start,end)


def load_or_encode(*,root,sid,allowed_sids,encoder,device,cache_root,metadata,
                   quota_bytes=T1Config().cache_bytes,resource_callback=None):
    _require_sid(sid,allowed_sids)
    if resource_callback: resource_callback()
    if metadata.get('sid')!=sid or metadata.get('encoder_sha256')!=encoder_hash(encoder):
        raise ValueError('cache encoder/session identity mismatch')
    raw_path=Path(root)/'cache/sessions'/f'{sid}.npz'
    if file_hash(raw_path)!=metadata.get('raw_sha256'): raise ValueError('raw checksum mismatch')
    key=cache_key(metadata); directory=Path(cache_root)
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'{key}.npz'; header=directory/f'{key}.json'
    if path.exists() or header.exists():
        if not path.is_file() or not header.is_file(): raise ValueError('partial token cache')
        saved=json.loads(header.read_text(encoding='utf-8'))
        if saved.get('metadata')!=metadata or saved.get('sha256')!=file_hash(path):
            raise ValueError('token cache checksum/identity mismatch')
        with np.load(path,allow_pickle=False) as data:
            return SessionTokens(sid,*(data[k] for k in ('center_ms','span_id','coverage','valid','embeddings')),
                                 int(data['start_ms']),int(data['end_ms']))
    tokens=encode_session(root,sid,allowed_sids,encoder,device,resource_callback=resource_callback)
    temporary=path.with_name(path.name+'.tmp'); temp_header=header.with_name(header.name+'.tmp')
    installed=False
    try:
        fields=asdict(tokens); fields.pop('sid')
        with temporary.open('wb') as f: np.savez_compressed(f,**fields)
        record={'metadata':metadata,'sha256':file_hash(temporary)}
        text=json.dumps(record,sort_keys=True,allow_nan=False)
        existing=sum(p.stat().st_size for p in directory.iterdir() if p.is_file() and p!=temporary)
        if existing+temporary.stat().st_size+len(text.encode())>quota_bytes:
            raise RuntimeError('token cache quota exceeded')
        temp_header.write_text(text,encoding='utf-8')
        temporary.replace(path); installed=True; temp_header.replace(header)
    except Exception:
        if installed and not header.exists(): path.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True); temp_header.unlink(missing_ok=True)
    return tokens


def audit_data(root,manifest):
    """Metadata-only audit; actual token coverage is recorded at encoding time."""
    rows=[]
    allowed=set(manifest['identities']['t24'])
    for sid,subject in sorted(manifest['subject_by_session'].items()):
        if subject not in allowed: raise ValueError('audit session outside T24')
        _require_sid(sid,{sid})
        path=Path(root)/'cache/sessions'/f'{sid}.npz'
        stat=path.stat()
        rows.append({'sid':sid,'subject':subject,'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns})
    return {'sessions':rows,'total_raw_bytes':sum(r['bytes'] for r in rows),
            'raw_arrays_decoded':False,'raw_bytes_read':False,'source_training_started':False,'outer_calls':0}
