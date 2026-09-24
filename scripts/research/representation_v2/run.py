"""Stage-gated CLI for representation research; P0 is read-only upstream."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("preflight")
    stage.add_argument("--plan", type=Path, default=DEFAULT_OUTPUT / "plan.json")
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(ROOT, args.plan.resolve().parent)
        print(json.dumps({"status": result["status"], "budget": result["budget"]}, sort_keys=True))


if __name__ == "__main__":
    main()
