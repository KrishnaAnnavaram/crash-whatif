"""Command line: ``crash-whatif <command>``. Run ``crash-whatif --help`` for the list."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from . import report as rpt
from . import study as st
from .config import ConfigError, Settings, load_dotenv
from .counterfactuals import FeatureSpace, RandomSearchCF, to_frame
from .schema import SchemaError, coerce, validate
from .synthetic import make_crashes

EXAMPLE_CRASH = {
    "make": "Mahindra", "vehicle_type": "pickup", "vehicle_year": 2012, "engine_type": "diesel",
    "engine_cc": 2500, "transmission": "manual", "cylinders": 4, "weight_kg": 1900, "length_mm": 4800,
    "width_mm": 1850, "height_mm": 1750, "safety_rating": 2, "airbags": 3, "abs": 1, "esc": 0,
    "tcs": 0, "tpms": 0, "location": "urban", "weather": "rain", "road_surface": "dry",
    "time_of_day": "night", "day_of_week": "Saturday", "driver_age": 58, "driver_gender": "Male",
}


def _raw(args, settings: Settings) -> tuple[pd.DataFrame, str]:
    path = getattr(args, "csv", None) or settings.data_path
    if path:
        return pd.read_csv(path), str(path)
    rows, mode = getattr(args, "rows", 6000), getattr(args, "mode", "signal")
    return make_crashes(rows, seed=settings.seed, mode=mode), f"synthetic ({rows} rows, mode {mode}, seed {settings.seed})"


def _study(args, settings: Settings, names=st.MODEL_NAMES) -> tuple[st.Study, str]:
    raw, source = _raw(args, settings)
    study = st.prepare(raw, settings)
    print(study.validation.summary())
    print(f"split (stratified, seed {settings.seed}): {study.split.sizes()}")
    return st.fit_all(study, names), source


def cmd_generate(args, settings):
    df = make_crashes(args.rows, seed=settings.seed, mode=args.mode)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows, mode {args.mode})")
    return 0


def cmd_validate(args, settings):
    _, report = validate(pd.read_csv(args.csv))
    print(report.summary())
    return 0


def cmd_train(args, settings):
    study, source = _study(args, settings)
    print(rpt.metrics_text(st.metrics_table(study)))
    out = Path(args.out or settings.out_dir) / "models"
    out.mkdir(parents=True, exist_ok=True)
    space = FeatureSpace.from_train(study.split.train, settings.actionable)
    for name, model in study.models.items():
        joblib.dump({"model": model, "space": space, "source": source}, out / f"{name}.joblib")
    print(f"saved {len(study.models)} models in {out}")
    return 0


def cmd_explain(args, settings):
    names = tuple(args.models.split(","))
    study, _ = _study(args, settings, names=names)
    print(rpt.importance_text(st.importance_tables(study, names=names, n_repeats=args.repeats), top=args.top))
    return 0


def cmd_audit(args, settings):
    study, _ = _study(args, settings, names=())
    print(json.dumps(rpt.jsonable(st.signal_audit(study)), indent=2))
    return 0


def cmd_whatif(args, settings):
    names = tuple(args.models.split(","))
    study, _ = _study(args, settings, names=names)
    bench = st.counterfactual_benchmark(study, names=names, n_queries=args.queries, n_cfs=args.cfs)
    print(rpt.benchmark_text(bench))
    if args.json:
        print(json.dumps(rpt.jsonable(bench), indent=2))
    return 0


def cmd_counterfactual(args, settings):
    bundle = joblib.load(args.model)
    model, space = bundle["model"], bundle["space"]
    records = json.loads(Path(args.input).read_text(encoding="utf-8") if args.input.endswith(".json") else args.input)
    query = coerce(pd.DataFrame([records])).iloc[0]
    p = float(model.proba(to_frame([query]))[0])
    print(f"model {model.name}: P(severe) = {p:.3f}, threshold {model.threshold:.2f}")
    result = RandomSearchCF(model, space, seed=settings.seed).generate(query, n_cfs=args.cfs)
    if not result.found:
        print("no counterfactual found inside the actionable columns and permitted ranges")
        return 0
    for i, (_, cf) in enumerate(result.counterfactuals.iterrows(), 1):
        changes = ", ".join(f"{c}: {query[c]} -> {cf[c]}" for c in space.changed(query, cf))
        print(f"{i}. {changes}  (P(severe) = {float(model.proba(result.counterfactuals.iloc[[i - 1]])[0]):.3f})")
    return 0


def cmd_report(args, settings):
    study, source = _study(args, settings)
    result = {
        "rows": study.split.sizes(),
        "class_counts": study.validation.class_counts,
        "settings": {"seed": settings.seed, "resample": settings.resample, "actionable": list(settings.actionable)},
        "metrics": st.metrics_table(study),
        "audit": st.signal_audit(study),
        "importance": st.importance_tables(study, n_repeats=args.repeats),
        "counterfactuals": st.counterfactual_benchmark(study, n_queries=args.queries),
        "three_class": st.three_class(study),
    }
    paths = rpt.write(result, args.out or settings.out_dir, source)
    print(rpt.metrics_text(result["metrics"]))
    print(rpt.importance_text(result["importance"]))
    print(rpt.benchmark_text(result["counterfactuals"]))
    print(f"wrote {paths['markdown']} and {paths['json']}")
    return 0


def cmd_demo(args, settings):
    print("crash-whatif offline demo: synthetic data, no download, no key, no network\n")
    study, _ = _study(args, settings)
    print("\n" + rpt.metrics_text(st.metrics_table(study)))
    audit = st.signal_audit(study)
    print(f"\nsignal audit: AUC make only {audit['auc_make_only']:.3f}, without make {audit['auc_without_make']:.3f}, "
          f"all columns {audit['auc_all_columns']:.3f}")
    print("\npermutation importance (drop of ROC AUC):")
    print(rpt.importance_text(st.importance_tables(study, n_repeats=5)))
    print("\ncounterfactual benchmark:")
    print(rpt.benchmark_text(st.counterfactual_benchmark(study, n_queries=args.queries)))
    model = study.models["random_forest"]
    query = coerce(pd.DataFrame([EXAMPLE_CRASH])).iloc[0]
    space = FeatureSpace.from_train(study.split.train, settings.actionable)
    result = RandomSearchCF(model, space, seed=settings.seed).generate(query, n_cfs=3)
    print(f"\nexample crash, random_forest P(severe) = {float(model.proba(to_frame([query]))[0]):.3f}")
    if not result.found:
        print("  no counterfactual found inside the actionable columns and permitted ranges")
    for i, (_, cf) in enumerate(result.counterfactuals.iterrows(), 1):
        print(f"  {i}. " + ", ".join(f"{c}: {query[c]} -> {cf[c]}" for c in space.changed(query, cf)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="crash-whatif", description="Crash severity models with counterfactual explanations.")
    sub = p.add_subparsers(dest="command", required=True)

    def data_args(c):
        c.add_argument("--csv", help="crash CSV (public column names or project names); default: synthetic")
        c.add_argument("--rows", type=int, default=6000, help="synthetic rows if no --csv")
        c.add_argument("--mode", choices=["signal", "make_rule"], default="signal", help="synthetic label mode")

    g = sub.add_parser("generate", help="write a synthetic crash CSV")
    g.add_argument("--out", default="data/crashes.csv")
    g.add_argument("--rows", type=int, default=6000)
    g.add_argument("--mode", choices=["signal", "make_rule"], default="signal")
    g.set_defaults(func=cmd_generate)

    v = sub.add_parser("validate", help="validate a crash CSV")
    v.add_argument("--csv", required=True)
    v.set_defaults(func=cmd_validate)

    t = sub.add_parser("train", help="fit all models and save them")
    data_args(t)
    t.add_argument("--out", help="output folder (default CRASH_WHATIF_OUT)")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("explain", help="permutation importance with repeats and a seed")
    data_args(e)
    e.add_argument("--models", default="logreg,random_forest")
    e.add_argument("--repeats", type=int, default=10)
    e.add_argument("--top", type=int, default=10)
    e.set_defaults(func=cmd_explain)

    a = sub.add_parser("audit", help="how much does make alone explain?")
    data_args(a)
    a.set_defaults(func=cmd_audit)

    w = sub.add_parser("whatif", help="counterfactual benchmark over N test rows")
    data_args(w)
    w.add_argument("--models", default=",".join(st.CF_MODELS))
    w.add_argument("--queries", type=int, default=None, help="default CRASH_WHATIF_N_QUERIES")
    w.add_argument("--cfs", type=int, default=3)
    w.add_argument("--json", action="store_true")
    w.set_defaults(func=cmd_whatif)

    c = sub.add_parser("counterfactual", help="counterfactuals for one crash with a saved model")
    c.add_argument("--model", required=True)
    c.add_argument("--input", required=True, help="JSON object, inline or a .json file")
    c.add_argument("--cfs", type=int, default=3)
    c.set_defaults(func=cmd_counterfactual)

    r = sub.add_parser("report", help="full study: metrics, audit, importance, benchmark, three classes")
    data_args(r)
    r.add_argument("--queries", type=int, default=None)
    r.add_argument("--repeats", type=int, default=10)
    r.add_argument("--out")
    r.set_defaults(func=cmd_report)

    d = sub.add_parser("demo", help="offline demo on synthetic data")
    data_args(d)
    d.add_argument("--queries", type=int, default=30)
    d.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        load_dotenv()
        settings = Settings.from_env()
        return args.func(args, settings)
    except (ConfigError, SchemaError, ValueError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
