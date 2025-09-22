viz_py = """import pathlib
import matplotlib.pyplot as plt

def plot_confusion_matrices(results, outdir, save=True):
fms = results["fold_metrics"]
for fm in fms:
cm = fm["confusion_matrix"]
fig, ax = plt.subplots(figsize=(4,4))
im = ax.imshow(cm, aspect="auto")
ax.set_title(f"Fold {fm['fold']} confusion matrix")
plt.colorbar(im, ax=ax)
if save:
p = pathlib.Path(outdir) / f"cm_fold{fm['fold']}.png"
fig.savefig(p, bbox_inches="tight", dpi=150)
plt.close(fig)

def plot_metric_distributions(results, outdir, save=True):
fms = results["fold_metrics"]
metrics = [k for k in fms[0].keys() if k not in ("confusion_matrix","fold")]
for m in metrics:
vals = [fm[m] for fm in fms]
fig, ax = plt.subplots(figsize=(4,3))
ax.hist(vals, bins=5)
ax.set_title(m)
if save:
p = pathlib.Path(outdir) / f"metric_{m}.png"
fig.savefig(p, bbox_inches="tight", dpi=150)
plt.close(fig)
"""
(src / "viz.py").write_text(viz_py, encoding="utf-8")
