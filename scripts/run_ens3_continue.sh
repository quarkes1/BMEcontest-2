#!/bin/bash
# 续链（去掉 d128 腿）：folds 2-4 补 r1/r2 → ens3 平均 → 解码 15m → exp_ens3/
# fold1 HARD_K=10 → exp_hardk10/。最后 canonical 恢复为 ens3。
set -e
cd "$(dirname "$0")/.."
source activate bme 2>/dev/null
OUT=outputs

for k in 2 3 4; do
  echo "===== fold $k r1 (no-ppg) ====="
  python -u scripts/train_ranker.py --fold $k --no-ppg 2>&1 | tail -2
  cp "$OUT/mm_ranker_fold${k}_val.npz" "$OUT/ens_r1_fold${k}.npz"
  echo "===== fold $k r2 (no-ppg) ====="
  python -u scripts/train_ranker.py --fold $k --no-ppg 2>&1 | tail -2
  cp "$OUT/mm_ranker_fold${k}_val.npz" "$OUT/ens_r2_fold${k}.npz"
done

echo "===== fold1 HARD_K=10 ====="
BME_HARD_K=10 python -u scripts/train_ranker.py --fold 1 --no-ppg 2>&1 | tail -2
cp "$OUT/mm_ranker_fold1_val.npz" "$OUT/hardk10_fold1.npz"

echo "===== ens3 平均 → 解码 15m ====="
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
    out = np.array([np.mean(keyed[(m["sid"], int(m["s"]), int(m["e"]))]) for m in metas])
    np.savez(OUT / f"mm_ranker_fold{k}_val.npz", score=out.astype(np.float32), meta=z["meta"])
    print(f"fold{k}: ens3 {len(out)} 候选")
EOF

mkdir -p "$OUT/exp_ens3"
for k in 0 1 2 3 4; do
  python -u scripts/rank_events_v2.py --fold $k --prior-grid 15m 2>&1 | tail -1
done
cp "$OUT/rank_events_v2_fold"*"_15m.json" "$OUT/exp_ens3/"
echo "===== summarize ens3 ====="
python scripts/summarize_v2.py 15m 2>&1 | tail -12

echo "===== fold1 hardk10 解码 ====="
mkdir -p "$OUT/exp_hardk10"
cp "$OUT/hardk10_fold1.npz" "$OUT/mm_ranker_fold1_val.npz"
python -u scripts/rank_events_v2.py --fold 1 --prior-grid 15m 2>&1 | tail -2
cp "$OUT/rank_events_v2_fold1_15m.json" "$OUT/exp_hardk10/"

echo "===== 恢复 canonical = ens3 ====="
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
echo "===== 续链完成 ====="
