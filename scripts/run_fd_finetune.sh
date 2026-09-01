#!/bin/bash
# 阶段二验收：FD 预训练权重 → 5 折微调 → 解码 → 官方评估对比（FD 预训练前后）
set -e
cd "$(dirname "$0")/.."
source activate bme 2>/dev/null
OUT=outputs

mkdir -p "$OUT/exp_fdpre"
for k in 0 1 2 3 4; do
  echo "===== fold $k FD 预训练微调 ====="
  python -u scripts/train_ranker.py --fold $k --no-ppg --init-from checkpoints/fd_pretrained_s1.pt 2>&1 | tail -2
  cp "$OUT/mm_ranker_fold${k}_val.npz" "$OUT/fdpre_fold${k}.npz"
done

echo "===== 解码 15m ====="
for k in 0 1 2 3 4; do
  python -u scripts/rank_events_v2.py --fold $k --prior-grid 15m 2>&1 | tail -1
done
cp "$OUT/rank_events_v2_fold"*"_15m.json" "$OUT/exp_fdpre/"

echo "===== 官方评估（FD 预训练前/后对比）====="
python - << 'EOF'
import sys, re, json
sys.path.insert(0, "scripts")
import official_iou_eval as oe
import src.config as config
import numpy as np

print(f"{'fold':>5} | {'FD预训练后 F1':>12} {'Sens':>6} {'PPV':>6} | {'ens3基线 F1':>12} {'Sens':>6} {'PPV':>6}")
f1_pre, f1_base = [], []
for k in range(5):
    j = json.loads((config.OUTPUT_DIR / f"rank_events_v2_fold{k}_15m.json").read_text(encoding="utf-8"))
    cfg_name = j["best"]["name"]
    val_rows, gate_prob, clf_pri, true_sid, _, _ = oe.v2.prepare_fold(k)
    mm = re.match(r"w([\d.]+)_t([\d.]+)_g([\d.]+)_p([\d.]+)_d(\d+)_k(\d+)", cfg_name)
    cfg = (float(mm[1]), float(mm[2]), float(mm[3]), float(mm[4]), int(mm[5]), int(mm[6]))
    m = oe.official_metrics(sum((oe.v2.decode_session(r, gate_prob, cfg, clf_pri)[0] for r in val_rows), []), true_sid)
    # ens3 基线：从备份恢复评估（用 archive 的 npz 分数）
    bz = np.load(config.OUTPUT_DIR / "archive_best_20260831" / f"mm_ranker_fold{k}_val.npz", allow_pickle=True)
    import numpy as _np
    bscore = bz["score"]
    bmeta = [json.loads(x.decode()) for x in bz["meta"]]
    dl = {(mm2["sid"], int(mm2["s"]), int(mm2["e"])): float(sc) for mm2, sc in zip(bmeta, bscore)}
    val_rows2 = []
    for r in val_rows:
        sid, act, sa, _, pri, sp = r
        va_scores = _np.full(len(act), _np.nan, _np.float32)
        for jj, c in enumerate(act):
            if (sid, c[0], c[1]) in dl:
                va_scores[jj] = dl[(sid, c[0], c[1])]
        val_rows2.append((sid, act, sa, va_scores, pri, sp))
    mb = oe.official_metrics(sum((oe.v2.decode_session(r, gate_prob, cfg, clf_pri)[0] for r in val_rows2), []), true_sid)
    f1_pre.append(m["f1"]); f1_base.append(mb["f1"])
    print(f"{k:>5} | {m['f1']:>12.3f} {m['sensitivity']:>6.3f} {m['ppv']:>6.3f} | "
          f"{mb['f1']:>12.3f} {mb['sensitivity']:>6.3f} {mb['ppv']:>6.3f}")
print(f"均值 | {np.mean(f1_pre):>12.3f} | {np.mean(f1_base):>12.3f}")
EOF

echo "===== 恢复 canonical = ens3 ====="
for k in 0 1 2 3 4; do
  cp "$OUT/archive_best_20260831/exp_ens3/rank_events_v2_fold${k}_15m.json" "$OUT/rank_events_v2_fold${k}_15m.json"
  cp "$OUT/archive_best_20260831/mm_ranker_fold${k}_val.npz" "$OUT/mm_ranker_fold${k}_val.npz"
done
echo "===== 完成 ====="
