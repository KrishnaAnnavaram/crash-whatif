"""Write the study as JSON and as a Markdown report with a short model card."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def metrics_text(rows: list[dict]) -> str:
    head = f"{'model':<19}{'thr':>5}{'ROC AUC':>9}{'95% CI':>17}{'PR AUC':>8}{'bal acc':>8}{'F1 mac':>8}{'Brier':>7}{'ECE':>7}"
    lines = [head]
    for r in rows:
        ci = r["roc_auc_ci95"]
        ci_text = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci[0] == ci[0] else "-"
        lines.append(
            f"{r['model']:<19}{r['threshold']:>5.2f}{r['roc_auc']:>9.3f}{ci_text:>17}{r['pr_auc_severe']:>8.3f}"
            f"{r['balanced_accuracy']:>8.3f}{r['f1_macro']:>8.3f}{r['brier']:>7.3f}{r['ece']:>7.3f}"
        )
    return "\n".join(lines)


def importance_text(tables: dict[str, list[dict]], top: int = 8) -> str:
    lines = []
    for name, rows in tables.items():
        items = ", ".join(f"{r['column']} {r['auc_drop']:+.3f}" for r in rows[:top])
        lines.append(f"{name}: {items}")
    return "\n".join(lines)


def benchmark_text(bench: dict) -> str:
    lines = [f"{'model':<19}{'queries':>8}{'found':>7}{'valid':>7}{'prox':>7}{'sparse':>8}{'divers':>8}{'plaus':>7}  top changed columns"]
    for name, b in bench.items():
        if not isinstance(b, dict) or "queries" not in b:
            continue
        g = lambda k: b[k]["mean"] if k in b else float("nan")  # noqa: E731
        top = ", ".join(list(b.get("changed_columns", {}))[:3])
        lines.append(
            f"{name:<19}{b['queries']:>8}{g('found'):>7.2f}{g('validity'):>7.2f}{g('proximity'):>7.2f}"
            f"{g('sparsity'):>8.2f}{g('diversity'):>8.2f}{g('plausibility'):>7.2f}  {top}"
        )
    return "\n".join(lines)


def write(study_result: dict, out_dir: str | Path, source: str) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "data_source": source}
    json_path = out / "report.json"
    md_path = out / "report.md"
    json_path.write_text(json.dumps(jsonable({**meta, **study_result}), indent=2), encoding="utf-8")
    synthetic = "synthetic" in source
    parts = [
        "# crash-whatif report",
        "",
        f"- Created (UTC): {meta['created_utc']}",
        f"- Data source: {source}" + (" (SYNTHETIC: not a finding about real crashes)" if synthetic else ""),
        f"- Rows: {study_result['rows']}, classes: {study_result['class_counts']}",
        f"- Resampling: {study_result['settings']['resample']}, seed {study_result['settings']['seed']}",
        "",
        "## Test metrics (real test rows only)",
        "",
        "```",
        metrics_text(study_result["metrics"]),
        "```",
        "",
        "## Signal audit",
        "",
        "```",
        json.dumps(jsonable({k: v for k, v in study_result["audit"].items() if k != "severity_share_by_make_train"}), indent=2),
        "```",
        "",
        "## Permutation importance (drop of ROC AUC, test rows)",
        "",
        "```",
        importance_text(study_result["importance"]),
        "```",
        "",
        "## Counterfactual benchmark",
        "",
        "```",
        benchmark_text(study_result["counterfactuals"]),
        "```",
        "",
        "## Intended use and limits",
        "",
        "- This is a study of model explanations. It is not a road-safety decision tool.",
        "- A counterfactual explains the MODEL. It does not prove a cause in the real world.",
        "- A human expert must review each conclusion. The data can be biased by where and how crashes are recorded.",
    ]
    md_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return {"json": json_path, "markdown": md_path}
