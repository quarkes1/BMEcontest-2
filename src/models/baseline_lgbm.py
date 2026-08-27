# -*- coding: utf-8 -*-
"""LightGBM 窗口分类基线：训练 + 阈值调优 + 滑窗概率 -> 事件列表。"""
import numpy as np
import lightgbm as lgb
import src.config as config

def train_one_fold(X, y, seed=config.RANDOM_SEED):
    """X/y 为窗口特征与标签；内部留 15% 分层样本做 early stopping。"""
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    val_idx = np.concatenate([
        rng.choice(pos_idx, size=max(1, int(len(pos_idx) * 0.15)), replace=False),
        rng.choice(neg_idx, size=max(1, int(len(neg_idx) * 0.15)), replace=False)])
    train_idx = np.setdiff1d(np.arange(len(y)), val_idx)
    ds = lgb.Dataset(X[train_idx], label=y[train_idx])
    dv = lgb.Dataset(X[val_idx], label=y[val_idx], reference=ds)
    params = {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
              "num_leaves": 63, "max_depth": 7, "feature_fraction": 0.8,
              "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1,
              "seed": seed}
    model = lgb.train(params, ds, num_boost_round=300, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
    return model

def _probs_to_events(probs, t0s, t1s, threshold):
    """滑窗概率 -> 事件列表（起止 ms，合并/过滤/边界膨胀）。"""
    on = probs >= threshold
    events = []
    i = 0
    while i < len(on):
        if on[i]:
            j = i
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            events.append((int(t0s[i]), int(t1s[j])))
            i = j + 1
        else:
            i += 1
    dil = config.BOUNDARY_DILATION_SEC * 1000
    merged = []
    for e in events:
        if merged and e[0] - merged[-1][1] <= config.EVENT_MERGE_GAP_SEC * 1000:
            merged[-1] = (merged[-1][0], e[1])
        else:
            merged.append(e)
    return [(max(0, s - dil), e + dil) for s, e in merged
            if (e - s) >= config.EVENT_MIN_DUR_SEC * 1000]

def predict_session(model, windows, threshold=0.5):
    """windows: list[dict]（含 feat/t0_ms/t1_ms）。返回 (事件列表, 概率序列)。"""
    X = [w["feat"] for w in windows]
    if not X:
        return [], []
    probs = model.predict(np.vstack(X))
    return (_probs_to_events(probs, [w["t0_ms"] for w in windows],
                             [w["t1_ms"] for w in windows], threshold), probs)
