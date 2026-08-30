#!/bin/bash
# 组合实验链（背景运行）：
#   1) 集成：r0(恢复run)+r1+r2 三跑 no-ppg 分数平均 → 解码 15m → exp_ens3/
#   2) 容量：d_model=128 单跑 → 解码 15m → exp_d128/
#   3) fold1 硬负样本探针：HARD_K=10 → 解码 fold1 → exp_hardk10/
# 最后把 ens3 平均分数恢复为 canonical mm_ranker_fold{k}_val.npz
set -e
cd "$(dirname "$0")/.."
source activate bme 2>/dev/null
OUT=outputs

# ---- 阶段 0：保存 r0（当前 canonical = 恢复 run） ----
for k in 0 1 2 3 4; do
  cp "$OUT/mm_ranker_fold${k}_val.npz" "$OUT/ens_r0_fold${k}.npz"
done

# ---- 阶段 1：每折 3 次训练（r1, r2 集成；d128 容量） ----
for k in 0 1 2 3 4; do
  echo "===== fold $k r1 (no-ppg) ====="
  python -u scripts/train_ranker.py --fold $k --no-ppg 2>&1 | tail -2
  cp "$OUT/mm_ranker_fold${k}_val.npz" "$OUT/ens_r1_fold${k}.npz"
  echo "===== fold $k r2 (no-ppg) ====="
  python -u scripts/train_ranker.py --fold $k --no-ppg 2>&1 | tail -2
  cp "$OUT/mm_ranker_fold${k}_val.npz" "$OUT/ens_r2_fold${k}.npz"
  echo "===== fold $k d128 ====="
  BME_D_MODEL=128 python -u scripts/train_ranker.py --fold $k --no-ppg 2>&1 | tail -2
  cp "$OUT/mm_ranker_fold${k}_val.npz" "$OUT/d128_fold${k}.npz"
done

# ---- 阶段 2：fold1 HARD_K=10 探针 ----
echo "===== fold1 HARD_K=10 ====="
BME_HARD_K=10 python -u scripts/train_ranker.py --fold 1 --no-ppg 2>&1 | tail -2
cp "$OUT/mm_ranker_fold1_val.npz" "$OUT/hardk10_fold1.npz"

# ---- 阶段 3：集成平均（按 (sid,s,e) 键匹配）→ 解码 15m ----
python - << 'EOF'
import json
import numpy as np
from pathlib import Path
OUT = Path("outputs")
for k in range(5):
    keyed = {}
    metas = None
    for r in ("r0", "r1", "r2"):
        z = np.load(OUT / f"ens_{r}_fold{k}.npz", allow_pickle=True)
        meta = [json.loads(x.decode()) for x in z["meta"]]
        if metas is None:
            metas = meta
        elif len(meta) != len(metas):
            raise SystemExit(f"meta len mismatch fold{k}: {len(meta)} vs {len(metas)}")
        for m, s in zip(meta, z["score"].astype(np.float64)):
            keyed.setdefault((m["sid"], int(m["s"]), int(m["e"])), []).append(s)
    vals = np.array([np.mean(v) for v in keyed.values()])
    keys = list(keyed.keys())
    # 保序写回：与 meta 相同顺序（keys 顺序 = 首 run 顺序，因 dict 按插入序）
    if metas is not None:
        out = np.array([keyed[(m["sid"], int(m["s"]), int(m["e"]))][0] for m in metas])
        assert len(out) == len(metas)
    else:
        out = vals
    np.savez(OUT / f"mm_ranker_fold{k}_val.npz", score=out.astype(np.float32), meta=z["meta"])
    print(f"fold{k}: ens3 {len(out)} 候选")
EOF

echo "===== 解码 ens3 ====="
mkdir -p "$OUT/exp_ens3"
for k in 0 1 2 3 4; do
  python -u scripts/rank_events_v2.py --fold $k --prior-grid 15m 2>&1 | tail -1
done
cp "$OUT/rank_events_v2_fold"*"_15m.json" "$OUT/exp_ens3/"
python scripts/summarize_v2.py 15m 2>&1 | tail -12

# ---- 阶段 4：d128 解码 ----
echo "===== 解码 d128 ====="
mkdir -p "$OUT/exp_d128"
for k in 0 1 2 3 4; do
  cp "$OUT/d128_fold${k}.npz" "$OUT/mm_ranker_fold${k}_val.npz"
  python -u scripts/rank_events_v2.py --fold $k --prior-grid 15m 2>&1 | tail -1
done
cp "$OUT/rank_events_v2_fold"*"_15m.json" "$OUT/exp_d128/"
python scripts/summarize_v2.py 15m 2>&1 | tail -12

# ---- 阶段 5：fold1 HARD_K=10 解码 ----
echo "===== 解码 hardk10 fold1 ====="
mkdir -p "$OUT/exp_hardk10"
cp "$OUT/hardk10_fold1.npz" "$OUT/mm_ranker_fold1_val.npz"
python -u scripts/rank_events_v2.py --fold 1 --prior-grid 15m 2>&1 | tail -2
cp "$OUT/rank_events_v2_fold1_15m.json" "$OUT/exp_hardk10/"

# ---- 阶段 6：恢复 canonical = ens3 ----
echo "===== 恢复 canonical ens3 ====="
for k in 0 1 2 3 4; do
  cp "$OUT/exp_ens3/rank_events_v2_fold${k}_15m.json" "$OUT/rank_events_v2_fold${k}_15m.json"
done
python - << 'EOF'
import json
import numpy as np
from pathlib import Path
OUT = Path("outputs")
for k in range(5):
    z = np.load(OUT / f"ens_r0_fold{k}.npz", allow_pickle=True)
    keyed = {}
    for r in ("r0", "r1", "r2"):
        zz = np.load(OUT / f"ens_{r}_fold{k}.npz", allow_pickle=True)
        meta = [json.loads(x.decode()) for x in zz["meta"]]
        for m, s in zip(meta, zz["score"].astype(np.float64)):
            keyed.setdefault((m["sid"], int(m["s"]), int(m["e"])), []).append(s)
    out = np.array([np.mean(keyed[(m["sid"], int(m["s"]), int(m["e"]))]) for m in
                    [json.loads(x.decode()) for x in z["meta"]]])
    np.savez(OUT / f"mm_ranker_fold{k}_val.npz", score=out.astype(np.float32), meta=z["meta"])
print("canonical restored to ens3")
EOF
echo "===== 全链完成 ====="
