"""Registered candidate-conditioned VICReg probe; no target labels consumed."""
from copy import deepcopy
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def rotate_yaw(values, angles):
    result = values.clone()
    c, s = angles.cos()[:, None], angles.sin()[:, None]
    for offset in (0, 3):
        result[:, offset] = c*values[:, offset] - s*values[:, offset+1]
        result[:, offset+1] = s*values[:, offset] + c*values[:, offset+1]
    return result


def augment(values, mask, center, scale):
    n = len(values)
    angles = (torch.rand(n, device=values.device)*2-1)*(torch.pi/18)
    x = rotate_yaw(values, angles) * (.9+.2*torch.rand(n,1,1,device=values.device))
    x = (x-center[None,:,None])/scale[None,:,None]
    x = x + .01*torch.randn_like(x)
    begins = torch.randint(0, 114, (n,1), device=values.device)
    positions = torch.arange(125, device=values.device)[None]
    keep = mask & ~((positions >= begins) & (positions < begins+12))
    return torch.where(keep[:,None,:], x, 0.), keep


def vicreg_loss(first, second):
    if first.shape != second.shape or first.ndim != 2 or len(first) < 2:
        raise ValueError('VICReg requires two aligned batches of at least two rows')
    invariance = F.mse_loss(first, second)
    variance = (F.relu(1-torch.sqrt(first.var(0)+1e-4)).mean() +
                F.relu(1-torch.sqrt(second.var(0)+1e-4)).mean())/2
    covariance = first.new_zeros(())
    for values in (first, second):
        centered = values-values.mean(0)
        cov = centered.T @ centered / (len(values)-1)
        off = cov - torch.diag_embed(cov.diagonal())
        covariance = covariance + off.square().sum()/values.shape[1]
    loss = 25*invariance+25*variance+covariance
    return loss, {'invariance': float(invariance.detach()), 'variance': float(variance.detach()),
                  'covariance': float(covariance.detach())}


def build_token_bank(bags, subjects, *, seed, max_per_subject=2048):
    if len(bags) != len(subjects) or not bags:
        raise ValueError('bank bags and subjects must align')
    choices, seen = {}, set()
    for bag, subject in sorted(zip(bags, subjects), key=lambda pair: pair[0].candidate_key):
        for index in np.flatnonzero(bag.token_mask):
            key = (bag.candidate_key[0], int(bag.token_center_ms[index]))
            if key not in seen:
                seen.add(key)
                choices.setdefault(str(subject), []).append((bag, index, key))
    rng = np.random.default_rng(seed)
    values, masks, owners, keys = [], [], [], []
    for subject, entries in sorted(choices.items()):
        keep = np.sort(rng.choice(len(entries), min(max_per_subject, len(entries)), replace=False))
        for index in keep:
            bag, row, key = entries[index]
            values.append(bag.values[row]); masks.append(bag.sample_mask[row])
            owners.append(subject); keys.append(key)
    if not values:
        raise ValueError('empty SSL bank')
    return {'values': np.asarray(values), 'masks': np.asarray(masks),
            'subjects': np.asarray(owners), 'keys': keys}


def fit_ssl(initial_encoder, bank, center, scale, *, seed=20260926, steps=5000,
            device='cpu', max_seconds=3600, progress=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    encoder = deepcopy(initial_encoder).to(device).train()
    projector = nn.Sequential(nn.Linear(16,64), nn.ReLU(), nn.Linear(64,64)).to(device)
    params = list(encoder.parameters()) + list(projector.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    center_t = torch.as_tensor(center, dtype=torch.float32, device=device)
    scale_t = torch.as_tensor(scale, dtype=torch.float32, device=device)
    names = sorted(set(bank['subjects']))
    indices = [np.flatnonzero(bank['subjects'] == name) for name in names]
    started = time.perf_counter()
    history, recent = [], []
    for step in range(1, steps+1):
        if time.perf_counter()-started > max_seconds:
            raise TimeoutError('SSL fixed-step budget incomplete')
        subjects = rng.integers(len(names), size=128)
        rows = np.array([rng.choice(indices[i]) for i in subjects])
        raw = torch.as_tensor(bank['values'][rows], dtype=torch.float32, device=device)
        mask = torch.as_tensor(bank['masks'][rows], device=device)
        first, m1 = augment(raw, mask, center_t, scale_t)
        second, m2 = augment(raw, mask, center_t, scale_t)
        optimizer.zero_grad(set_to_none=True)
        loss, parts = vicreg_loss(projector(encoder(first,m1)), projector(encoder(second,m2)))
        if not torch.isfinite(loss):
            raise RuntimeError('non-finite SSL objective')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1.)
        if not torch.isfinite(norm):
            raise RuntimeError('non-finite SSL gradient')
        optimizer.step()
        recent.append([float(loss.detach()), parts['invariance'], parts['variance'], parts['covariance'], float(norm)])
        if step % 100 == 0 or step == steps:
            means = np.mean(recent,axis=0)
            record = dict(zip(('loss','invariance','variance','covariance','gradient_norm'),map(float,means)))
            record['step'] = step
            record['elapsed_seconds'] = time.perf_counter()-started
            history.append(record); recent=[]
            if progress is not None:
                progress(record)
    encoder.eval()
    vectors = []
    take = np.linspace(0,len(bank['values'])-1,min(4096,len(bank['values'])),dtype=int)
    with torch.inference_mode():
        for left in range(0,len(take),512):
            rows = take[left:left+512]
            mask = torch.as_tensor(bank['masks'][rows],device=device)
            x = torch.as_tensor(bank['values'][rows],dtype=torch.float32,device=device)
            x = torch.where(mask[:,None,:],(x-center_t[None,:,None])/scale_t[None,:,None],0.)
            vectors.append(encoder(x,mask).cpu().numpy())
    z = np.concatenate(vectors).astype(np.float64)
    std = z.std(0)
    eigen = np.maximum(np.linalg.eigvalsh(np.cov(z,rowvar=False)),0)
    fractions = eigen / max(float(eigen.sum()),1e-20)
    positive = fractions[fractions>0]
    rank = float(np.exp(-np.sum(positive*np.log(positive)))) if len(positive) else 0.
    collapse = {'std':std.tolist(),'mean_std':float(std.mean()),'dimensions_std_ge_0001':int((std>=.001).sum()),
                'effective_rank':rank,'passed':bool(std.mean()>=.01 and (std>=.001).sum()>=8 and rank>=2)}
    return {'encoder':encoder,'projector':projector.eval(),'history':history,'collapse':collapse,
            'steps':steps,'wall_seconds':time.perf_counter()-started}
