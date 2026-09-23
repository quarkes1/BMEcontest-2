#!/usr/bin/env python
"""Run the sealed, data-first challenger sprint without fitting or scoring a model."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research.data_sprint import (  # noqa: E402
    ANALYSIS_RATE_HZ,
    SCALES_SECONDS,
    SCHEMA_VERSION,
    disagreement_counts,
    disagreement_fold_support,
    expected_window_coverage,
    git_sha,
    periodicity_metrics,
    prepare_output_dir,
    robust_mad,
    select_truth_level_fn,
    sha256_file,
    sha256_json,
    temporal_metrics,
)
from src.data.manifests import load_meals  # noqa: E402


def _native(value):
    if isinstance(value, dict):
        return {str(k): _native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_native(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(path: Path, title: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([f"# {title}", "", *lines, ""]) + "\n", encoding="utf-8")


class SignalStore:
    def __init__(self, root: Path):
        self.root = root
        self.cache: dict[str, dict] = {}

    def load(self, session_id: str) -> dict:
        if session_id not in self.cache:
            path = self.root / "cache" / "sessions" / f"{session_id}.npz"
            if not path.exists():
                raise FileNotFoundError(f"required sealed session cache is missing: {path}")
            with np.load(path, allow_pickle=False) as z:
                rate = float(z["row_rate"])
                step = max(1, int(round(rate / ANALYSIS_RATE_HZ)))
                valid = z["imu_valid"][::step].astype(bool)
                self.cache[session_id] = {
                    "acc": z["acc"][:, ::step].astype(np.float64),
                    "gyro": z["gyro"][:, ::step].astype(np.float64),
                    "time": z["t_acc"][::step].astype(np.int64),
                    "valid": valid,
                    "rate": rate / step,
                    "source_rate": rate,
                    "rows": int(z["rows"]),
                }
        return self.cache[session_id]

    def window(self, session_id: str, centre_ms: int, seconds: int) -> tuple[np.ndarray, np.ndarray, float, float]:
        data = self.load(session_id)
        half = seconds * 500
        mask = (data["time"] >= centre_ms - half) & (data["time"] <= centre_ms + half) & data["valid"]
        coverage = expected_window_coverage(int(np.sum(mask)), data["rate"], seconds)
        return data["acc"][:, mask], data["gyro"][:, mask], data["rate"], coverage

    def profile(self, session_id: str, centre_ms: int, seconds: int = 60, bins: int = 120) -> np.ndarray:
        acc, gyro, _, _ = self.window(session_id, centre_ms, seconds)
        if acc.shape[1] < 4:
            return np.full((2, bins), np.nan)
        values = np.vstack([np.linalg.norm(acc, axis=0), np.linalg.norm(gyro, axis=0)])
        old = np.linspace(0, 1, values.shape[1])
        new = np.linspace(0, 1, bins)
        return np.vstack([np.interp(new, old, channel) for channel in values])

    def session_summary(self, session_id: str) -> dict[str, float]:
        data = self.load(session_id)
        valid = data["valid"]
        acc = data["acc"][:, valid]
        gyro = data["gyro"][:, valid]
        metrics = temporal_metrics(acc, gyro, data["rate"])
        t = data["time"]
        unique_t = np.unique(t[valid])
        gaps = np.diff(unique_t)
        positive = gaps[gaps > 0]
        typical = np.median(positive) if len(positive) else np.nan
        metrics.update(
            sensor_coverage=float(np.mean(valid)),
            gap_count=int(np.sum(positive > max(2000.0, 5 * typical))) if np.isfinite(typical) else 0,
            session_duration_h=float((t[-1] - t[0]) / 3_600_000) if len(t) else np.nan,
        )
        return metrics


def populations(root: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    base = root / "outputs" / "challenger_70" / "e04_verifier_audit"
    tp = pd.read_csv(base / "tp_reference.csv")
    fp = pd.read_csv(base / "fp_reference.csv")
    fn_pairs = pd.read_csv(base / "reachable_fn_candidates.csv")
    fn = select_truth_level_fn(fn_pairs)
    atlas = pd.read_csv(root / "outputs" / "challenger_70" / "error_atlas" / "fn.csv")
    c0 = atlas.loc[atlas["fn_stage"] == "FN-C0"].copy()
    c0["fold"] = c0["outer_fold"]
    c0["candidate_key"] = c0["truth_key"]
    c0["blend_score"] = np.nan
    groups = {"matched_tp": tp, "final_fp": fp, "reachable_fn": fn, "c0_fn": c0}
    combined = []
    for group, frame in groups.items():
        frame = frame.copy()
        frame["population"] = group
        combined.append(frame)
    return groups, pd.concat(combined, ignore_index=True, sort=False)


def track_a(groups: dict[str, pd.DataFrame], output: Path) -> dict:
    target = prepare_output_dir(output / "e04_1")
    eligible = {k: v for k, v in groups.items() if k != "c0_fn"}
    totals = {name: disagreement_counts(frame) for name, frame in eligible.items()}
    by_fold = {
        name: {str(int(fold)): disagreement_counts(part) for fold, part in frame.groupby("fold")}
        for name, frame in eligible.items()
    }
    fold_support = disagreement_fold_support(by_fold)
    write_json(target / "direction_counts.json", {"threshold": 0.20, "populations": totals})
    write_json(target / "direction_by_fold.json", {"threshold": 0.20, "populations": by_fold})
    fn_direction = totals["reachable_fn"]
    fp_direction = totals["final_fp"]
    useful = (
        max(fn_direction["lr_minus_lgbm_ge_0_20"], fn_direction["lgbm_minus_lr_ge_0_20"]) >= 5
        and fold_support["stable"]
    )
    write_report(target / "report.md", "E04.1 — Frozen LR/LGBM disagreement direction", [
        "Diagnostic only. Scores, threshold, models, and populations are frozen; no F1 was computed.",
        "",
        "| Population | N | LR−LGBM ≥ .20 | LGBM−LR ≥ .20 | Neither |",
        "|---|---:|---:|---:|---:|",
        *[f"| {name} | {v['population_size']} | {v['lr_minus_lgbm_ge_0_20']} | {v['lgbm_minus_lr_ge_0_20']} | {v['neither']} |" for name, v in totals.items()],
        "",
        f"Reachable-FN LR-high share exceeds FP in {fold_support['fn_lr_excess_fold_count']}/{fold_support['comparable_fold_count']} folds.",
        f"Evidence flag for later adaptive-fusion consideration: **{'yes' if useful else 'no'}**. This is not a challenger result.",
    ])
    return {"counts": totals, "by_fold": by_fold, "fold_support": fold_support, "adaptive_fusion_signal": useful}


def _group_summary(frame: pd.DataFrame, metric_names: list[str]) -> dict:
    result = {}
    for group, part in frame.groupby("population"):
        result[group] = {name: {"median": float(part[name].median()), "mad": robust_mad(part[name].to_numpy()), "n": int(part[name].notna().sum())} for name in metric_names}
    return result


def _fold_direction(frame: pd.DataFrame, metric: str, a: str = "matched_tp", b: str = "final_fp") -> dict:
    rows = {}
    for fold in range(5):
        left = frame[(frame.population == a) & (frame.fold == fold)][metric].median()
        right = frame[(frame.population == b) & (frame.fold == fold)][metric].median()
        rows[str(fold)] = {"a_median": float(left), "b_median": float(right), "difference": float(left - right)}
    signs = [np.sign(v["difference"]) for v in rows.values() if np.isfinite(v["difference"]) and v["difference"] != 0]
    consistency = max(signs.count(1), signs.count(-1)) if signs else 0
    return {"folds": rows, "consistent_fold_count": int(consistency)}


def extract_metrics(combined: pd.DataFrame, store: SignalStore) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    scale_rows, periodic_rows, profiles = [], [], []
    for _, row in combined.iterrows():
        centre = int((float(row.start_ms) + float(row.end_ms)) / 2)
        identity = {
            "population": row.population,
            "fold": int(row.fold),
            "session_id": row.session_id,
            "subject_id": row.subject_id,
            "candidate_key": row.candidate_key,
            "start_ms": int(row.start_ms),
            "end_ms": int(row.end_ms),
            "duration_s": float((row.end_ms - row.start_ms) / 1000),
            "blend_score": float(row.blend_score) if pd.notna(row.blend_score) else np.nan,
        }
        for seconds in SCALES_SECONDS:
            acc, gyro, rate, coverage = store.window(row.session_id, centre, seconds)
            metrics = temporal_metrics(acc, gyro, rate)
            scale_rows.append({**identity, "scale_s": seconds, "window_coverage": coverage, **metrics})
        acc, gyro, rate, coverage = store.window(row.session_id, centre, 60)
        if acc.shape[1] >= 4:
            acc_mag = np.linalg.norm(acc, axis=0)
            gyro_mag = np.linalg.norm(gyro, axis=0)
            acc_z = (acc_mag - np.median(acc_mag)) / (robust_mad(acc_mag) + 1e-9)
            gyro_z = (gyro_mag - np.median(gyro_mag)) / (robust_mad(gyro_mag) + 1e-9)
            periodic = periodicity_metrics(acc_z + gyro_z, rate)
        else:
            periodic = periodicity_metrics(np.array([]), rate)
        periodic_rows.append({**identity, "window_coverage": coverage, **periodic})
        profiles.append(store.profile(row.session_id, centre))
    return pd.DataFrame(scale_rows), pd.DataFrame(periodic_rows), np.asarray(profiles)


def temporal_outputs(scale: pd.DataFrame, output: Path) -> dict:
    target = prepare_output_dir(output / "temporal_scale")
    scale.to_csv(target / "metrics.csv", index=False)
    metrics = ["activity_magnitude", "acc_variance", "gyro_variance", "short_term_energy", "sign_change_rate", "peak_count", "burst_count", "active_fraction", "local_contrast"]
    summaries, directions = {}, {}
    candidate_scores = []
    for seconds in SCALES_SECONDS:
        part = scale[scale.scale_s == seconds]
        summaries[str(seconds)] = _group_summary(part, metrics)
        directions[str(seconds)] = {metric: _fold_direction(part, metric) for metric in metrics}
        for metric in metrics:
            values = summaries[str(seconds)]
            tp, fp, fn = values["matched_tp"][metric]["median"], values["final_fp"][metric]["median"], values["reachable_fn"][metric]["median"]
            separation = abs(tp - fp) / (values["matched_tp"][metric]["mad"] + values["final_fp"][metric]["mad"] + 1e-9)
            fn_tp_like = abs(fn - tp) < abs(fn - fp)
            candidate_scores.append((directions[str(seconds)][metric]["consistent_fold_count"], separation, fn_tp_like, seconds, metric))
    best = max(candidate_scores, key=lambda x: (x[0], x[1], x[2]))
    signal = bool(best[0] >= 4 and best[1] >= 0.25 and best[2])
    result = {"population_summaries": summaries, "fold_directions": directions, "best_candidate": {"scale_s": best[3], "metric": best[4], "consistent_folds": best[0], "robust_separation": best[1], "reachable_fn_tp_like": best[2]}, "temporal_scale_signal": signal}
    result["coverage_audit"] = {
        "rows_below_0_8": int(np.sum(scale.window_coverage < 0.8)),
        "total_rows": int(len(scale)),
        "minimum": float(scale.window_coverage.min()),
        "note": "low-coverage rows remain explicit and yield NaN proxies when fewer than four samples are available",
    }
    write_json(target / "summary.json", result)
    write_report(target / "report.md", "Temporal-scale audit", [
        "Fixed scales: 5/10/15/30/60 s. All measurements use cached raw IMU and the fixed incumbent populations.",
        "",
        f"Strongest registered proxy: `{best[4]}` at {best[3]} s; TP–FP direction agreed in {best[0]}/5 folds, robust separation={best[1]:.3f}, reachable FN TP-like={best[2]}.",
        f"Coverage audit: {result['coverage_audit']['rows_below_0_8']}/{result['coverage_audit']['total_rows']} scale rows are below 0.8 coverage; no imputation was used.",
        f"Mechanism flag: **{'present' if signal else 'not established'}**. No feature or classifier was promoted.",
    ])
    return result


def periodic_outputs(periodic: pd.DataFrame, output: Path) -> dict:
    target = prepare_output_dir(output / "periodicity")
    periodic.to_csv(target / "metrics.csv", index=False)
    metrics = ["autocorrelation_first_peak_s", "autocorrelation_peak_strength", "dominant_frequency_hz", "spectral_entropy", "burst_interval_mean_s", "burst_interval_variance_s2", "burst_count", "repetition_regularity", "peak_interval_mean_s", "peak_interval_variance_s2"]
    summaries = _group_summary(periodic, metrics)
    directions = {metric: _fold_direction(periodic, metric) for metric in metrics}
    candidates = []
    for metric in metrics:
        tp, fp, fn = summaries["matched_tp"][metric]["median"], summaries["final_fp"][metric]["median"], summaries["reachable_fn"][metric]["median"]
        separation = abs(tp - fp) / (summaries["matched_tp"][metric]["mad"] + summaries["final_fp"][metric]["mad"] + 1e-9)
        candidates.append((directions[metric]["consistent_fold_count"], separation, abs(fn - tp) < abs(fn - fp), metric))
    best = max(candidates, key=lambda x: (x[0], x[1], x[2]))
    signal = bool(best[0] >= 4 and best[1] >= 0.25 and best[2])
    result = {"population_summaries": summaries, "fold_directions": directions, "best_candidate": {"metric": best[3], "consistent_folds": best[0], "robust_separation": best[1], "reachable_fn_tp_like": best[2]}, "periodicity_signal": signal}
    write_json(target / "summary.json", result)
    write_report(target / "report.md", "Repetition / periodicity audit", [
        "Computed on one fixed 60 s local window; no model was trained.", "",
        f"Strongest proxy: `{best[3]}`; TP–FP direction agreed in {best[0]}/5 folds, robust separation={best[1]:.3f}, reachable FN TP-like={best[2]}.",
        f"Mechanism flag: **{'present' if signal else 'not established'}**.",
    ])
    return result


def morphology_outputs(combined: pd.DataFrame, profiles: np.ndarray, output: Path) -> dict:
    target = prepare_output_dir(output / "temporal_morphology")
    np.savez_compressed(target / "profiles.npz", profiles=profiles.astype(np.float32), population=combined.population.to_numpy(str), fold=combined.fold.to_numpy(int))
    group_profiles = {}
    for group in combined.population.unique():
        values = profiles[combined.population.to_numpy() == group]
        group_profiles[group] = {"mean": np.nanmean(values, axis=0).tolist(), "q25": np.nanquantile(values, .25, axis=0).tolist(), "q75": np.nanquantile(values, .75, axis=0).tolist(), "n": int(len(values))}
    write_json(target / "profile_summary.json", {"channels": ["acc_magnitude", "gyro_magnitude"], "window_seconds": 60, "bins": 120, "groups": group_profiles, "probability_streams": "unavailable without rerunning canonical extraction; intentionally omitted"})
    flat = profiles.reshape(len(profiles), -1)
    column_medians = np.nanmedian(flat, axis=0)
    flat = np.where(np.isfinite(flat), flat, column_medians)
    scaled = StandardScaler().fit_transform(flat)
    embedding = PCA(n_components=3, random_state=0).fit_transform(scaled)
    pca_frame = pd.DataFrame({"population": combined.population, "fold": combined.fold, "pc1": embedding[:, 0], "pc2": embedding[:, 1], "pc3": embedding[:, 2]})
    centroids = pca_frame.groupby("population")[["pc1", "pc2", "pc3"]].median().to_dict("index")
    pca = PCA(n_components=3, random_state=0).fit(scaled)
    result = {"explained_variance_ratio": pca.explained_variance_ratio_.tolist(), "population_centroids": centroids, "interpretation": "descriptive_only_no_classifier"}
    write_json(target / "pca_summary.json", result)
    write_report(target / "report.md", "Temporal morphology audit", [
        "Fixed 60 s ACC/GYRO magnitude profiles were downsampled to 120 bins. PCA is descriptive only.", "",
        f"First three components explain {sum(result['explained_variance_ratio']):.1%} of standardized profile variance.",
        "The compact PCA centroids are insufficient on their own to justify a temporal encoder; fold-stable separation must agree with the scale/periodicity audits.",
    ])
    return result


def subject_outputs(scale: pd.DataFrame, combined: pd.DataFrame, store: SignalStore, output: Path) -> dict:
    target = prepare_output_dir(output / "subject_relative")
    session_rows = []
    identity = combined[["session_id", "subject_id", "fold"]].drop_duplicates()
    for _, row in identity.iterrows():
        session_rows.append({"session_id": row.session_id, "subject_id": row.subject_id, "fold": int(row.fold), **store.session_summary(row.session_id)})
    sessions = pd.DataFrame(session_rows)
    subject = sessions.groupby("subject_id").agg(activity_median=("activity_magnitude", "median"), activity_mad=("activity_magnitude", lambda x: robust_mad(x.to_numpy())), energy_baseline=("short_term_energy", "median"), gyro_baseline=("gyro_variance", "median"), session_count=("session_id", "nunique")).reset_index()
    candidates = scale[scale.scale_s == 60].merge(subject, on="subject_id", how="left")
    candidates["candidate_activity_ratio"] = candidates.activity_magnitude / (candidates.activity_median + 1e-9)
    candidates["candidate_energy_ratio"] = candidates.short_term_energy / (candidates.energy_baseline + 1e-9)
    candidates["candidate_gyro_ratio"] = candidates.gyro_variance / (candidates.gyro_baseline + 1e-9)
    candidates["candidate_activity_percentile"] = candidates.groupby("subject_id")["activity_magnitude"].rank(pct=True)
    metrics = ["candidate_activity_ratio", "candidate_energy_ratio", "candidate_gyro_ratio", "candidate_activity_percentile"]
    result = {"population_summaries": _group_summary(candidates, metrics), "fold_directions": {m: _fold_direction(candidates, m) for m in metrics}, "baseline_scope": "all sealed cached sessions referenced by fixed populations", "subjects": int(subject.subject_id.nunique()), "sessions": int(len(sessions))}
    weak = candidates[candidates.fold.isin([0, 2])].candidate_activity_ratio.median()
    strong = candidates[~candidates.fold.isin([0, 2])].candidate_activity_ratio.median()
    result["weak_vs_other_activity_ratio"] = {"weak_median": float(weak), "other_median": float(strong), "ratio": float(weak / (strong + 1e-9))}
    result["subject_shift_signal"] = bool(abs(np.log((weak + 1e-9) / (strong + 1e-9))) >= np.log(1.25))
    candidates.to_csv(target / "metrics.csv", index=False)
    write_json(target / "summary.json", result)
    write_report(target / "report.md", "Subject-relative audit", [
        f"Baseline coverage: {result['subjects']} subjects / {result['sessions']} referenced sealed sessions.", "",
        f"Weak-fold versus other-fold candidate activity ratio: {result['weak_vs_other_activity_ratio']['ratio']:.3f}.",
        f"Subject-shift flag: **{'present' if result['subject_shift_signal'] else 'not established'}**.",
    ])
    return result


def hand_scenario_outputs(combined: pd.DataFrame, scale: pd.DataFrame, periodic: pd.DataFrame, output: Path) -> dict:
    target = prepare_output_dir(output / "hand_scenario")
    meals = load_meals()
    rows = []
    for _, row in combined.iterrows():
        centre = (row.start_ms + row.end_ms) / 2
        match = meals[(meals.externalid == row.subject_id) & (meals.before_ms <= centre) & (meals.after_ms >= centre)]
        rows.append({"candidate_key": row.candidate_key, "population": row.population, "fold": int(row.fold), "subject_id": row.subject_id, "scene": match.iloc[0].scene if len(match) else "unknown", "duration_s": (row.end_ms - row.start_ms) / 1000, "blend_score": row.blend_score})
    frame = pd.DataFrame(rows)
    signal = scale[scale.scale_s == 60][["candidate_key", "activity_magnitude", "short_term_energy", "gyro_variance"]]
    repetition = periodic[["candidate_key", "repetition_regularity"]]
    frame = frame.merge(signal, on="candidate_key", how="left").merge(repetition, on="candidate_key", how="left")
    counts = frame.groupby(["population", "scene"]).size().unstack(fill_value=0).to_dict("index")
    fold_counts = frame.groupby(["fold", "scene"]).size().unstack(fill_value=0).to_dict("index")
    known = frame[frame.scene != "unknown"]
    metrics = ["duration_s", "blend_score", "activity_magnitude", "short_term_energy", "gyro_variance", "repetition_regularity"]
    scene_metrics = {
        scene: {metric: {"median": float(part[metric].median()), "n": int(part[metric].notna().sum())} for metric in metrics}
        for scene, part in known.groupby("scene")
    }
    result = {"population_scene_counts": counts, "fold_scene_counts": fold_counts, "scene_metric_summaries": scene_metrics, "known_scene_fraction": float(np.mean(frame.scene != "unknown")), "limitation": "scene is assigned only when the analysed interval centre overlaps labelled meal metadata; unmatched FP scene remains unknown"}
    frame.to_csv(target / "metrics.csv", index=False)
    write_json(target / "summary.json", result)
    write_report(target / "report.md", "Hand / scenario audit", [
        f"Known scene coverage is {result['known_scene_fraction']:.1%}.", "",
        "FP hand/scenario cannot be assigned reliably from the available metadata, so no subgroup mechanism is claimed. Unknown values were retained rather than inferred.",
    ])
    return result


def fold_outputs(root: Path, combined: pd.DataFrame, scale: pd.DataFrame, store: SignalStore, output: Path) -> dict:
    target = prepare_output_dir(output / "fold_weakness")
    sixty = scale[scale.scale_s == 60]
    all_fn = pd.read_csv(root / "outputs" / "challenger_70" / "error_atlas" / "fn.csv")
    rows = []
    for fold in range(5):
        pop = combined[combined.fold == fold]
        sessions = pop.session_id.unique()
        summaries = [store.session_summary(s) for s in sessions]
        rows.append({
            "fold": fold,
            "subjects": int(pop.subject_id.nunique()), "sessions": int(len(sessions)),
            "matched_tp": int(np.sum(pop.population == "matched_tp")), "final_fp": int(np.sum(pop.population == "final_fp")),
            "eligible_meal_count": int(np.sum(pop.population == "matched_tp") + np.sum(all_fn.outer_fold == fold)),
            "reachable_fn": int(np.sum(pop.population == "reachable_fn")), "c0_fn": int(np.sum(pop.population == "c0_fn")),
            "short_event_fraction": float(np.mean(pop.duration_s < 600)) if "duration_s" in pop else float(np.mean((pop.end_ms-pop.start_ms) < 600000)),
            "median_event_duration_s": float(np.median((pop.end_ms-pop.start_ms)/1000)),
            "median_candidate_activity": float(sixty[sixty.fold == fold].activity_magnitude.median()),
            "median_sensor_coverage": float(np.median([s["sensor_coverage"] for s in summaries])),
            "gap_count": int(sum(s["gap_count"] for s in summaries)),
            "fp_density_per_session": float(np.sum(pop.population == "final_fp") / max(len(sessions), 1)),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(target / "metrics.csv", index=False)
    weak = frame[frame.fold.isin([0, 2])].mean(numeric_only=True)
    other = frame[~frame.fold.isin([0, 2])].mean(numeric_only=True)
    ratios = {name: float(weak[name] / (other[name] + 1e-9)) for name in ["median_candidate_activity", "median_event_duration_s", "short_event_fraction", "fp_density_per_session", "median_sensor_coverage"]}
    notable = {k: v for k, v in ratios.items() if v < 0.8 or v > 1.25}
    result = {"by_fold": frame.to_dict("records"), "weak_fold_ratios_vs_others": ratios, "notable_shared_differences": notable, "candidate_volume": "not available as a complete ledger; fixed-population event volume reported instead"}
    write_json(target / "summary.json", result)
    write_report(target / "report.md", "Fold weakness audit", [
        "Fold 0/2 were compared only on cached data properties; no fold-specific policy was evaluated.", "",
        f"Shared differences beyond the predeclared 0.8–1.25 descriptive band: `{json.dumps(_native(notable), ensure_ascii=False)}`.",
    ])
    return result


def fp_outputs(scale: pd.DataFrame, periodic: pd.DataFrame, output: Path) -> dict:
    target = prepare_output_dir(output / "fp_structure")
    fp = scale[(scale.population == "final_fp") & (scale.scale_s == 60)].merge(
        periodic[["candidate_key", "repetition_regularity", "dominant_frequency_hz", "burst_count"]], on="candidate_key", suffixes=("", "_periodic")
    )
    activity_hi = scale[(scale.population == "final_fp") & (scale.scale_s == 60)].activity_magnitude.median()
    modes = []
    for _, row in fp.iterrows():
        if row.duration_s < 120:
            mode = "short_burst"
        elif row.repetition_regularity >= 0.7 and row.burst_count_periodic >= 3:
            mode = "periodic_repetitive"
        elif row.activity_magnitude >= activity_hi and row.active_fraction >= 0.20:
            mode = "long_continuous_movement"
        elif row.burst_count >= 4 and row.active_fraction < 0.20:
            mode = "fragmented_high_activity"
        elif row.dominant_frequency_hz < 0.25:
            mode = "low_frequency_long_movement"
        else:
            mode = "unresolved_mixed"
        modes.append(mode)
    fp["behavior_mode"] = modes
    counts = Counter(modes)
    by_fold = fp.groupby(["fold", "behavior_mode"]).size().unstack(fill_value=0).to_dict("index")
    dominant = [(name, count) for name, count in counts.most_common() if count >= 10 and fp[fp.behavior_mode == name].fold.nunique() >= 3]
    result = {"counts": dict(counts), "by_fold": by_fold, "dominant_cross_fold_modes": dominant, "taxonomy": "fixed deterministic descriptive rules; not a classifier"}
    fp.to_csv(target / "metrics.csv", index=False)
    write_json(target / "summary.json", result)
    write_report(target / "report.md", "FP behavioral structure", [
        "The taxonomy uses fixed, interpretable timing rules and is descriptive only.", "",
        f"Cross-fold modes with at least 10 FP across at least 3 folds: `{json.dumps(dominant, ensure_ascii=False)}`.",
    ])
    return result


def choose_status(track_a_result: dict, temporal: dict, periodic: dict, subject: dict, morphology: dict) -> tuple[str, str]:
    signals = []
    if track_a_result["adaptive_fusion_signal"]:
        signals.append(("ADAPTIVE_FUSION_CANDIDATE", "frozen LR/LGBM direction"))
    if temporal["temporal_scale_signal"]:
        signals.append(("TEMPORAL_SCALE_CANDIDATE", "fold-stable short-scale proxy"))
    if periodic["periodicity_signal"]:
        signals.append(("PERIODICITY_CANDIDATE", "fold-stable repetition proxy"))
    if subject["subject_shift_signal"]:
        signals.append(("SUBJECT_NORMALIZATION_CANDIDATE", "weak-fold subject-relative shift"))
    if len(signals) > 1:
        return "MULTIPLE_SIGNALS_FOUND_NEEDS_HUMAN_SELECTION", "; ".join(v for _, v in signals)
    if signals:
        return signals[0]
    return "NO_NEW_SIGNAL_FOUND", "none of the predeclared descriptive gates was met"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=Path("outputs/challenger_70/data_sprint"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    groups, combined = populations(root)
    track_a_result = track_a(groups, output)
    store = SignalStore(root)
    scale, periodic, profiles = extract_metrics(combined, store)
    temporal_result = temporal_outputs(scale, output)
    periodic_result = periodic_outputs(periodic, output)
    morphology_result = morphology_outputs(combined, profiles, output)
    subject_result = subject_outputs(scale, combined, store, output)
    hand_result = hand_scenario_outputs(combined, scale, periodic, output)
    fold_result = fold_outputs(root, combined, scale, store, output)
    fp_result = fp_outputs(scale, periodic, output)
    status, rationale = choose_status(track_a_result, temporal_result, periodic_result, subject_result, morphology_result)

    fold_files = sorted((root / "cache" / "splits").glob("fold*.json"))
    e04 = root / "outputs" / "challenger_70" / "e04_verifier_audit"
    incumbent = json.loads((root / "release" / "event_stack_incumbent.json").read_text(encoding="utf-8"))
    manifest = {
        "diagnostic_only": True, "training_performed": False, "outer_f1_computed": False,
        "status": status, "status_rationale": rationale, "source_commit_sha": git_sha(root),
        "incumbent_run_key": incumbent["run_key"],
        "fold_assignment_checksum": sha256_json({p.name: sha256_file(p) for p in fold_files}),
        "feature_schema_version": SCHEMA_VERSION,
        "candidate_policy_version": f"incumbent:{incumbent['run_key']}",
        "eligible_denominator_checksum": incumbent["diagnostic_set_sha256"],
        "matcher_version": "iou-event-0.25-canonical",
        "population_sizes": {k: int(len(v)) for k, v in groups.items()},
        "sampling_units": {"matched_tp": "final matched event", "final_fp": "unmatched final event", "reachable_fn": "unique GT with highest frozen-blend reachable candidate", "c0_fn": "GT event"},
        "source_evidence": {name: sha256_file(e04 / name) for name in ["reachable_fn_candidates.csv", "tp_reference.csv", "fp_reference.csv", "manifest.json"]},
        "analysis_rate_hz": ANALYSIS_RATE_HZ, "scales_seconds": list(SCALES_SECONDS),
        "headline_unchanged": {"f1": incumbent["f1"], "tp": 114, "eligible_truth": 153, "predictions": 197},
    }
    write_json(output / "manifest.json", manifest)
    summary = [
        "# Challenger data characterization sprint", "",
        "> Diagnostic only: no training, threshold tuning, policy change, outer F1, release, README headline, or dist change.", "",
        f"**End state: `{status}`** — {rationale}.", "",
        "The official headline remains `0.6514285714 / 114 / 153 / 197`.", "",
        "## Required decisions", "",
        f"1. Adaptive fusion: **{'retain as a Tier 1 candidate' if track_a_result['adaptive_fusion_signal'] else 'stop'}**.",
        f"2. New temporal-scale signal: **{'yes' if temporal_result['temporal_scale_signal'] else 'not established'}**; strongest proxy `{temporal_result['best_candidate']['metric']}` at {temporal_result['best_candidate']['scale_s']} s.",
        f"3. Repetition/periodicity signal: **{'yes' if periodic_result['periodicity_signal'] else 'not established'}**; strongest proxy `{periodic_result['best_candidate']['metric']}`.",
        "4. Temporal morphology: PCA remains descriptive and does not independently justify a sequence model.",
        f"5. Subject-relative shift: **{'yes' if subject_result['subject_shift_signal'] else 'not established'}**.",
        f"6. Fold 0/2 shared mechanism: descriptive outliers are `{json.dumps(_native(fold_result['notable_shared_differences']), ensure_ascii=False)}`; no fold-specific intervention is permitted.",
        f"7. Dominant FP mode: `{json.dumps(_native(fp_result['dominant_cross_fold_modes']), ensure_ascii=False)}`.",
        f"8. Next challenger mechanism: **{status}**; human review is required before any implementation.",
        "9. Stop directions: any signal that failed its predeclared fold-stability/robust-separation gate; no further outer-derived mining.",
        f"10. Representation re-extraction: **{'likely required for temporal candidate' if temporal_result['temporal_scale_signal'] or periodic_result['periodicity_signal'] else 'not justified by this sprint'}**.",
        "",
        "## Population contract", "",
        "Matched TP: 114 final events; final FP: 83 events; reachable FN: 31 unique GT (highest frozen-score witness from 32 pairs); C0 FN: 6 GT. All folds and scores are frozen incumbent evidence.",
    ]
    (output / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": status, "cached_sessions_loaded": len(store.cache)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
