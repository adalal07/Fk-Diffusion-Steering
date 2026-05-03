import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any


LEGACY_FIELDS = (
    "exact_match_score",
    "character_accuracy",
    "legibility_score",
    "completeness_score",
    "extra_text_penalty",
    "layout_coherence",
)


@dataclass
class ParsedRow:
    raw: dict[str, Any]
    parsed_json: dict[str, Any] | None
    reward: float | None
    target_text: str
    sampling_idx: int | None
    particle_idx: int | None


def _to_float(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _legacy_formula(pj: dict[str, Any]) -> float | None:
    vals = [_to_float(pj.get(k)) for k in LEGACY_FIELDS]
    if any(v is None for v in vals):
        return None
    em, ca, lg, cp, ep, lc = vals
    score = 0.40 * em + 0.25 * ca + 0.15 * lg + 0.10 * cp + 0.10 * lc - 0.20 * ep
    return max(0.0, min(1.0, float(score)))


def _looks_like_full_prompt(s: str) -> bool:
    return len(s.split()) >= 6


def _load_rows(path: str) -> list[ParsedRow]:
    rows: list[ParsedRow] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pj = rec.get("parsed_json")
            if not isinstance(pj, dict):
                pj = None
            rows.append(
                ParsedRow(
                    raw=rec,
                    parsed_json=pj,
                    reward=_to_float(rec.get("reward")),
                    target_text=str(rec.get("target_text", "")),
                    sampling_idx=(
                        int(rec["sampling_idx"])
                        if rec.get("sampling_idx") is not None
                        else None
                    ),
                    particle_idx=(
                        int(rec["particle_idx"])
                        if rec.get("particle_idx") is not None
                        else None
                    ),
                )
            )
    return rows


def _write_csv(path: str, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def analyze(rows: list[ParsedRow], mismatch_threshold: float) -> dict[str, Any]:
    n = len(rows)
    parsed_rows = [r for r in rows if r.parsed_json is not None]
    rewards = [r.reward for r in rows if r.reward is not None]

    # per-target stats
    by_target: dict[str, list[ParsedRow]] = defaultdict(list)
    for r in rows:
        by_target[r.target_text].append(r)

    per_target = []
    for t, rr in by_target.items():
        rr_rewards = [x.reward for x in rr if x.reward is not None]
        per_target.append(
            {
                "target_text": t,
                "count": len(rr),
                "mean_reward": (sum(rr_rewards) / len(rr_rewards)) if rr_rewards else None,
                "zero_reward_count": sum(1 for x in rr_rewards if x == 0.0),
                "looks_like_full_prompt": _looks_like_full_prompt(t),
            }
        )

    # formula mismatch
    mismatches = []
    abs_diffs = []
    for r in parsed_rows:
        if r.reward is None:
            continue
        calc = _legacy_formula(r.parsed_json)
        if calc is None:
            continue
        d = abs(calc - r.reward)
        abs_diffs.append(d)
        if d > mismatch_threshold:
            pj = r.parsed_json
            mismatches.append(
                {
                    "sampling_idx": r.sampling_idx,
                    "particle_idx": r.particle_idx,
                    "target_text": r.target_text,
                    "detected_text": str(pj.get("detected_text", "")),
                    "logged_reward": r.reward,
                    "calc_reward": calc,
                    "abs_diff": d,
                    "short_reason": str(pj.get("short_reason", "")),
                }
            )

    # contradiction: reason claims no text/illegible but detected_text non-empty
    contradictions = []
    for r in parsed_rows:
        pj = r.parsed_json
        dt = str(pj.get("detected_text", "")).strip()
        reason = str(pj.get("short_reason", "")).lower()
        if dt and (
            "no text" in reason
            or "not detected" in reason
            or "illegible" in reason
            or "unreadable" in reason
        ):
            contradictions.append(
                {
                    "sampling_idx": r.sampling_idx,
                    "particle_idx": r.particle_idx,
                    "target_text": r.target_text,
                    "detected_text": dt,
                    "reward": r.reward,
                    "short_reason": str(pj.get("short_reason", "")),
                }
            )

    # saturation / repeated templates
    tuple_counter: Counter[tuple[Any, ...]] = Counter()
    for r in parsed_rows:
        pj = r.parsed_json
        tpl = (
            round(float(pj.get("exact_match_score", -1)), 3),
            round(float(pj.get("character_accuracy", -1)), 3),
            round(float(pj.get("legibility_score", -1)), 3),
            round(float(pj.get("completeness_score", -1)), 3),
            round(float(pj.get("extra_text_penalty", -1)), 3),
            round(float(pj.get("layout_coherence", -1)), 3),
            round(float(r.reward), 3) if r.reward is not None else None,
        )
        tuple_counter[tpl] += 1
    top_templates = [
        {"count": c, "score_tuple": list(k)} for k, c in tuple_counter.most_common(10)
    ]

    return {
        "n_rows": n,
        "parsed_json_rate": (len(parsed_rows) / n) if n else 0.0,
        "reward_mean": (sum(rewards) / len(rewards)) if rewards else None,
        "reward_min": min(rewards) if rewards else None,
        "reward_max": max(rewards) if rewards else None,
        "n_targets": len(by_target),
        "n_targets_looking_like_full_prompt": sum(
            1 for t in by_target if _looks_like_full_prompt(t)
        ),
        "formula_mismatch_mean_abs_diff": (
            sum(abs_diffs) / len(abs_diffs) if abs_diffs else None
        ),
        "formula_mismatch_count_over_threshold": len(mismatches),
        "formula_mismatch_threshold": mismatch_threshold,
        "contradiction_count": len(contradictions),
        "per_target": per_target,
        "top_repeated_score_templates": top_templates,
        "mismatches": mismatches,
        "contradictions": contradictions,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline failure-mode analysis for VLM OCR logs.")
    ap.add_argument("--log-path", required=True, help="Path to vlm_ocr_intermediate_logs.jsonl")
    ap.add_argument(
        "--output-dir",
        default="output/vlm_ocr_analysis",
        help="Directory to write analysis artifacts",
    )
    ap.add_argument(
        "--mismatch-threshold",
        type=float,
        default=0.03,
        help="Absolute diff threshold for logged reward vs recomputed legacy formula",
    )
    args = ap.parse_args()

    rows = _load_rows(args.log_path)
    report = analyze(rows, mismatch_threshold=float(args.mismatch_threshold))
    os.makedirs(args.output_dir, exist_ok=True)

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _write_csv(
        os.path.join(args.output_dir, "mismatches.csv"),
        [
            "sampling_idx",
            "particle_idx",
            "target_text",
            "detected_text",
            "logged_reward",
            "calc_reward",
            "abs_diff",
            "short_reason",
        ],
        report["mismatches"],
    )
    _write_csv(
        os.path.join(args.output_dir, "contradictions.csv"),
        [
            "sampling_idx",
            "particle_idx",
            "target_text",
            "detected_text",
            "reward",
            "short_reason",
        ],
        report["contradictions"],
    )
    _write_csv(
        os.path.join(args.output_dir, "per_target_stats.csv"),
        ["target_text", "count", "mean_reward", "zero_reward_count", "looks_like_full_prompt"],
        report["per_target"],
    )

    # human-readable markdown
    md_path = os.path.join(args.output_dir, "report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# VLM OCR Failure-Mode Report\n\n")
        f.write(f"- log_path: `{args.log_path}`\n")
        f.write(f"- rows: {report['n_rows']}\n")
        f.write(f"- parsed_json_rate: {report['parsed_json_rate']:.4f}\n")
        f.write(
            f"- reward mean/min/max: {report['reward_mean']:.4f} / "
            f"{report['reward_min']:.4f} / {report['reward_max']:.4f}\n"
            if report["reward_mean"] is not None
            else "- reward stats unavailable\n"
        )
        f.write(f"- targets: {report['n_targets']}\n")
        f.write(
            f"- targets looking like full prompt: {report['n_targets_looking_like_full_prompt']}\n"
        )
        f.write(
            "- formula mismatch mean abs diff: "
            f"{report['formula_mismatch_mean_abs_diff']:.4f}\n"
            if report["formula_mismatch_mean_abs_diff"] is not None
            else "- formula mismatch mean abs diff unavailable\n"
        )
        f.write(
            f"- mismatches over threshold ({report['formula_mismatch_threshold']}): "
            f"{report['formula_mismatch_count_over_threshold']}\n"
        )
        f.write(f"- contradiction count: {report['contradiction_count']}\n\n")

        f.write("## Top Repeated Score Templates\n\n")
        for tpl in report["top_repeated_score_templates"]:
            f.write(f"- count={tpl['count']}: `{tpl['score_tuple']}`\n")

        f.write("\n## Artifacts\n\n")
        f.write("- `summary.json`\n")
        f.write("- `mismatches.csv`\n")
        f.write("- `contradictions.csv`\n")
        f.write("- `per_target_stats.csv`\n")

    print(f"Wrote analysis to: {args.output_dir}")
    print(f"- {summary_path}")
    print(f"- {md_path}")


if __name__ == "__main__":
    main()

