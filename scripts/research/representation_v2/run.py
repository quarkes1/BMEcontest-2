"""Stage-gated CLI for representation research; P0 is read-only upstream."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.data.manifests import load_sensor_index
from src.pipeline.event_stack import EventRef
from scripts.research.representation_v2.dataset import PreprocessConfig, build_token_bag, prepare_session
from scripts.research.representation_v2.preflight import audit


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "outputs/challenger_70/representation_v2"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def preflight(root: Path, output: Path) -> dict[str, object]:
    """Record a P0 audit without loading validation arrays or fitting models."""
    result = audit(root)
    output.mkdir(parents=True, exist_ok=True)
    plan = {
        "experiment": "RV2-P0-001",
        "base_git_sha": result["base_git_sha"],
        "spec": "docs/challenge_070突破任务书_v2.md",
        "status": "APPROVED_FOR_P0_TO_P3_SUBJECT_TO_GATES",
        "outer_calls": 0,
        "release_changes": False,
        "schema": "representation-v2-20260924",
        "seed_primary": 20260924,
        "seed_confirmation": 20260925,
        "token_seconds": 5,
        "stride_seconds": 5,
        "sampling_hz": 25,
        "token_samples": 125,
        "event_bins": 64,
        "target_steps_per_fit": 2_000,
        "fd_pretrain_steps": 10_000,
        "source_dataset": "doi:10.48804/CN8VBB v1 FD-I",
        "target_arms": ["Q0_random_frozen", "Q1_scratch", "Q2_fd_frozen", "Q3_fd_finetuned"],
        "old_fd_checkpoint_reusable": False,
        "budget": result["budget"],
        "source_rules": [
            "outer train only",
            "no E01/E03.5/E04/Data Sprint/R1 row tables in training",
            "every learned ancestor excludes its held-out subject",
        ],
    }
    evidence = {
        "evidence_type": "historical diagnostic audit only",
        "not_reusable_for_current_gate": True,
        "items": [
            {"id": "R1_population", "status": "confirmed", "source": "scripts/research/temporal_rep/run_r1.py:26-33",
             "finding": "Population is selected from E04 outer-derived matched TP, final FP and reachable FN."},
            {"id": "R1_unordered", "status": "confirmed", "source": "scripts/research/temporal_rep/run_r1.py:63-67",
             "finding": "Top-k token embedding mean is invariant to event token order."},
            {"id": "R1_center_and_time", "status": "confirmed", "source": "scripts/research/temporal_rep/run_r1.py:37-44",
             "finding": "Only central 120 seconds are sampled and interpolation uses row index rather than timestamps."},
            {"id": "R1_training_budget", "status": "confirmed", "source": "scripts/research/temporal_rep/run_r1.py:78-83",
             "finding": "Thirty epochs correspond to thirty full-batch optimizer updates per fold; convergence was not established."},
            {"id": "R1_candidate_metric", "status": "confirmed", "source": "outputs/challenger_70/temporal_rep/cache/manifest.json",
             "finding": "145 positive and 83 negative selected rows; always-positive candidate F1 = 290/373 = 0.77748."},
            {"id": "E06_meta_dependencies", "status": "confirmed", "source": "scripts/research/run_tier1_challengers.py:185-203",
             "finding": "Base OOF scores and meta OOF share the same group split without fully nesting base-model training inside meta training."},
            {"id": "E05_policy_reselection", "status": "confirmed", "source": "scripts/research/run_tier1_challengers.py:185-200",
             "finding": "The augmented arm refits LR/LGBM and reselects the incumbent policy grid; churn is not a frozen-score effect."},
            {"id": "FD_legacy_checkpoint", "status": "confirmed", "source": "checkpoints/fd_pretrained_s1.pt",
             "finding": "Legacy checkpoint provenance is insufficient for this protocol and safe weights-only loading fails."},
        ],
    }
    _write_json(output / "plan.json", plan)
    _write_json(output / "evidence_review.json", evidence)
    _write_json(output / "budget.json", result["budget"])
    _write_json(output / "manifests/p0_audit.json", result)
    _write_json(output / "state.json", {"status": result["status"], "outer_calls": 0, "stage": "P0"})
    (output / "protocol.md").write_text(
        "# Representation v2 protocol\n\n"
        "This research uses only current outer-train subjects at each wrapper. "
        "Every learned score must carry a transitive train-subject ancestry excluding its held-out subject. "
        "Existing sealed-outer diagnostics are historical context and never training rows. "
        "Inner scores are selection evidence. Full outer and release changes require separate human approval.\n",
        encoding="utf-8",
    )
    (output / "summary.md").write_text(
        "# Representation v2 status\n\n"
        f"P0 status: `{result['status']}`. "
        "No model has been fitted and no outer evaluation has been run. "
        "The old FD checkpoint is excluded; FD-I source data is present. "
        "Proceed to P1 only after confirming the listed input and resource assumptions.\n",
        encoding="utf-8",
    )
    return result


def input_audit(root: Path, output: Path) -> dict[str, object]:
    """P1 smoke only: sample session caches from outer-fold-0 training subjects."""
    state_path = output / "state.json"
    if not state_path.is_file() or json.loads(state_path.read_text(encoding="utf-8")).get("status") != "P0_READY":
        raise RuntimeError("P1 input audit requires P0_READY")
    split = json.loads((root / "cache/splits/fold0.json").read_text(encoding="utf-8"))
    index = load_sensor_index()
    session_subjects = dict(zip(index.session_id.astype(str), index.externalid.astype(str)))
    chosen: dict[str, str] = {}
    for sid in sorted(split["train_sessions"]):
        chosen.setdefault(session_subjects[sid], sid)
    rows = []
    started = time.perf_counter()
    for subject, sid in list(sorted(chosen.items()))[:20]:
        path = root / "cache/sessions" / f"{sid}.npz"
        with np.load(path, allow_pickle=False) as payload:
            session = SimpleNamespace(**{key: payload[key] for key in ("acc", "gyro", "t_acc", "imu_valid", "row_rate")})
        valid = session.imu_valid
        stamps = session.t_acc[valid]
        prepared = prepare_session(session)
        start = int(prepared.first_ms)
        end = min(int(prepared.last_ms), start + 5_000)
        token = build_token_bag(prepared, EventRef(sid, start, end), PreprocessConfig())
        rows.append({
            "subject_id": subject, "session_id": sid, "row_rate": float(session.row_rate),
            "spans": len(prepared.spans), "valid_rows": int(valid.sum()),
            "repeated_stamp_fraction": float(np.mean(np.diff(stamps) == 0)) if len(stamps) > 1 else None,
            "first_token_coverage": float(token.coverage[0]),
            "first_token_valid": bool(token.token_mask[0]),
            "acc_median_norm_raw": float(np.median(np.linalg.norm(session.acc[:, valid], axis=0))),
            "gyro_abs_p99_raw": float(np.percentile(np.abs(session.gyro[:, valid]), 99)),
        })
    if not rows:
        raise RuntimeError("no eligible outer-train sessions for P1 input audit")
    result = {
        "stage": "P1_INPUT_AUDIT", "status": "P1_READY", "outer_validation_accessible": False,
        "source_role": "fold0 train sessions only; no labels or GT loaded",
        "sampled_subjects": len(rows), "wall_seconds": time.perf_counter() - started,
        "packet_timestamp_expansion": "anchor plus ordered row offsets, bounded by next anchor",
        "target_units": "ACC/GYRO raw counts; no physical conversion in the scratch path",
        "fd_acc_units_observed": "m/s^2 from a read-only FD-I participant sample",
        "fd_gyro_units": "unverified against target raw gyro counts",
        "fd_transfer_status": "FD_SOURCE_UNIT_REVIEW_REQUIRED",
        "rows": rows,
    }
    _write_json(output / "p1_input_audit/input_audit.json", result)
    _write_json(state_path, {"status": "P1_READY", "stage": "P1", "outer_calls": 0,
                             "fd_transfer_status": result["fd_transfer_status"]})
    with (output / "summary.md").open("a", encoding="utf-8") as handle:
        handle.write(f"\nP1 sampled {len(rows)} fold-0 training subjects in {result['wall_seconds']:.2f} s. "
                     "The target cache uses packet timestamps, valid spans and raw sensor counts; FD-I has "
                     "physical-scale acceleration. Target-only training is input-ready. FD unit alignment "
                     "still requires review before a transfer arm.\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("preflight")
    stage.add_argument("--plan", type=Path, default=DEFAULT_OUTPUT / "plan.json")
    cache_stage = commands.add_parser("cache")
    cache_stage.add_argument("--plan", type=Path, default=DEFAULT_OUTPUT / "plan.json")
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(ROOT, args.plan.resolve().parent)
        print(json.dumps({"status": result["status"], "budget": result["budget"]}, sort_keys=True))
    elif args.command == "cache":
        result = input_audit(ROOT, args.plan.resolve().parent)
        print(json.dumps({"status": result["status"], "sampled_subjects": result["sampled_subjects"],
                          "fd_transfer_status": result["fd_transfer_status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
