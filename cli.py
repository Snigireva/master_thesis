import argparse, json, sys, time, pathlib, yaml
from . import io, models, viz

def _load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("_runtime", {})["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return cfg

def cmd_fit(cfg):
    df = io.load_table(cfg)
    X, y, meta = io.make_features(df, cfg)
    results = models.cross_validated_fit(X, y, meta, cfg)
    outdir = pathlib.Path(cfg["output"]["dir"]); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "cv_results.json").write_text(json.dumps(results, indent=2))
    (outdir / "run_summary.json").write_text(json.dumps({"config": cfg}, indent=2))
    print("Saved:", outdir / "cv_results.json")

def cmd_plot(cfg):
    outdir = pathlib.Path(cfg["output"]["dir"]) ; outdir.mkdir(parents=True, exist_ok=True)
    results = json.loads((outdir / "cv_results.json").read_text())
    viz.plot_confusion_matrices(results, outdir, save=cfg["output"].get("save_plots", True))
    viz.plot_metric_distributions(results, outdir, save=cfg["output"].get("save_plots", True))
    print("Plots saved to", outdir)

def main(argv=None):
    p = argparse.ArgumentParser(prog="ms-lesion-ml", description="MRI lesion pipeline (rare-disease friendly)")
    p.add_argument("--config", required=True, help="Path to YAML config")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fit")
    sub.add_parser("plot")
    args = p.parse_args(argv)
    cfg = _load_config(args.config)
    if args.cmd == "fit":
        cmd_fit(cfg)
    elif args.cmd == "plot":
        cmd_plot(cfg)

if __name__ == "__main__":
    main()
