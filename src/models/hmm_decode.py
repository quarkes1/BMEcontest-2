# -*- coding: utf-8 -*-
"""L4 状态解码（spec §3）：2 态 HMM（静息/进食），Viterbi 全会话解码。
发射 = 校准后的融合分；转移自环 0.98（进食）/ 0.995（静息）。
部署时纯 numpy（W3 ONNX 打包不包含本模块的 torch 依赖——本模块本就无 torch）。"""
import numpy as np

STATE_REST, STATE_EAT = 0, 1
LOG_A = np.log(np.array([[0.995, 0.005],
                         [0.02, 0.98]]))          # 转移矩阵（对数域）


def calibrate(probs, pos_frac=0.05):
    """分位数校准：把分数映射到经验进食概率（部署用训练集统计；此处用自助法近似）。"""
    q = np.quantile(probs, 1.0 - pos_frac)
    return np.clip(probs / (q + 1e-9), 0, 1)


def viterbi(probs, sharp=2.0):
    """2 态 Viterbi 解码。probs: (n,) 进食概率。返回 (n,) 状态序列（0/1）。
    sharp: 发射概率锐化指数（>1 提升置信度区分）。"""
    n = len(probs)
    p_eat = np.clip(probs, 1e-6, 1 - 1e-6)
    emit = np.stack([np.log(1 - p_eat), np.log(p_eat)])        # (2, n)，经 sharp 修正
    if sharp != 1.0:
        emit = np.stack([np.log(1 - p_eat ** sharp), np.log(p_eat ** sharp)])
    dp = np.full((2, n), -np.inf)
    back = np.zeros((2, n), dtype=np.int8)
    dp[:, 0] = np.log(np.array([0.5, 0.5])) + emit[:, 0]
    for t in range(1, n):
        for s in range(2):
            cand = dp[:, t - 1] + LOG_A[:, s]
            best = int(np.argmax(cand))
            dp[s, t] = cand[best] + emit[s, t]
            back[s, t] = best
    states = np.empty(n, dtype=np.int8)
    states[-1] = int(np.argmax(dp[:, -1]))
    for t in range(n - 2, -1, -1):
        states[t] = back[states[t + 1], t + 1]
    return states


def decode_events(states, t0_ms, t1_ms,
                  merge_gap_s=30, min_dur_s=45, dilation_s=6):
    """状态序列 → 事件列表（进食段合并/过滤/膨胀，与 events.py 口径一致）。"""
    segs = []
    cur = None
    for i in range(len(states)):
        if states[i] == STATE_EAT:
            if cur is None:
                cur = [int(t0_ms[i]), int(t1_ms[i])]
            else:
                cur[1] = int(t1_ms[i])
        elif cur is not None:
            segs.append(cur); cur = None
    if cur is not None:
        segs.append(cur)
    merged = []
    for s, e in segs:
        if merged and s - merged[-1][1] <= merge_gap_s * 1000:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    out = []
    for s, e in merged:
        if (e - s) < min_dur_s * 1000:
            continue
        out.append((max(0, s - dilation_s * 1000), e + dilation_s * 1000))
    return out
