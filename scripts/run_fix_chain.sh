#!/bin/bash
# 时间轴修复全链：cand 6ch 全量重建（新 env 时间轴）→ 5 折重训 → 解码 → 评估
set -e
cd "$(dirname "$0")/.."
source activate bme 2>/dev/null
OUT=outputs

# cand_windows 全删重建（旧候选基于漂移时间轴）
for k in 0 1 2 3 4; do
  echo "===== 重建 cand_windows fold $k（真实时间轴） ====="
  python -c "
from pathlib import Path
d = Path('cache/cand_windows/fold$k')
n = sum(1 for p in d.glob('*.npz') if p.unlink() or True)
print(f'  清除 {n}')"
  BME_NO_GYRO=1 python -u scripts/build_candidate_windows.py --fold $k > "$OUT/fix_cw_f${k}.log" 2>&1 || { echo "fold$k 失败"; tail -3 "$OUT/fix_cw_f${k}.log"; exit 1; }
  tail -1 "$OUT/fix_cw_f${k}.log"
done

# 5 折重训
for k in 0 1 2 3 4; do
  echo "===== fold $k 重训 ====="
  python -u scripts/train_ranker.py --fold $k --no-ppg --init-from checkpoints/fd_pretrained_s1.pt \
    > "$OUT/fix_train_f${k}.log" 2>&1 || { echo "fold$k 失败"; tail -3 "$OUT/fix_train_f${k}.log"; exit 1; }
  cp "$OUT/mm_ranker_fold${k}_val.npz" "$OUT/fix_fold${k}.npz"
  tail -1 "$OUT/fix_train_f${k}.log"
done

# 解码 + 评估
for k in 0 1 2 3 4; do
  python -u scripts/rank_events_v2.py --fold $k --prior-grid 15m > "$OUT/fix_decode_f${k}.log" 2>&1 || exit 1
done
python - << 'EOF'
import sys, re, json
sys.path.insert(0, "scripts")
import official_iou_eval as oe
import src.config as config
import numpy as np
f1s = []
print(f"{'fold':>4} | {'F1':>6} {'Sens':>6} {'PPV':>6} | cfg")
for k in range(5):
    j = json.loads((config.OUTPUT_DIR / f"rank_events_v2_fold{k}_15m.json").read_text(encoding="utf-8"))
    cfg_name = j["best"]["name"]
    val_rows, gate_prob, clf_pri, true_sid, _, _ = oe.v2.prepare_fold(k)
    mm = re.match(r"w([\d.]+)_t([\d.]+)_g([\d.]+)_p([\d.]+)_d(\d+)_k(\d+)", cfg_name)
    cfg = (float(mm[1]), float(mm[2]), float(mm[3]), float(mm[4]), int(mm[5]), int(mm[6]))
    m = oe.official_metrics(sum((oe.v2.decode_session(r, gate_prob, cfg, clf_pri)[0] for r in val_rows), []), true_sid)
    f1s.append(m["f1"])
    print(f"{k:>4} | {m['f1']:>6.3f} {m['sensitivity']:>6.3f} {m['ppv']:>6.3f} | {cfg_name}")
print(f"均值 F1 = {np.mean(f1s):.3f}（修复前 0.319）")
EOF
echo "===== 修复链完成 ====="
