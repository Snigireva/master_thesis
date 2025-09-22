from typing import Dict, Any
import numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (balanced_accuracy_score, recall_score,
                             precision_score, confusion_matrix)
from sklearn.naive_bayes import GaussianNB, MultinomialNB

try:
    from imblearn.over_sampling import RandomOverSampler
except Exception:
    RandomOverSampler = None

def _specificity_macro(y_true, y_pred, labels=None) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    spes = []
    for k in range(cm.shape[0]):
        tn = cm.sum() - (cm[k,:].sum() + cm[:,k].sum() - cm[k,k])
        fp = cm[:,k].sum() - cm[k,k]
        spes.append( tn / (tn + fp) if (tn + fp) > 0 else 0.0 )
    return float(np.mean(spes))

def _metrics(y_true, y_pred, labels=None) -> Dict[str, float]:
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity_macro": float(recall_score(y_true, y_pred, average="macro")),
        "specificity_macro": float(_specificity_macro(y_true, y_pred, labels)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

def _get_model(name: str):
    if name == "gaussian_nb":
        return GaussianNB()
    if name == "multinomial_nb":
        return MultinomialNB()
    raise NotImplementedError(f"Model '{name}' not implemented")

def cross_validated_fit(X, y, meta, cfg) -> Dict[str, Any]:
    rng = int(cfg["cv"]["random_state"])
    k = int(cfg["cv"]["folds"])
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=rng)
    model_name = cfg["model"]["type"]
    oversample = bool(cfg["model"].get("oversample", False))
    results = defaultdict(list)
    labels = np.unique(y)

    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        Xtr, ytr = X[tr], y[tr]
        Xte, yte = X[te], y[te]

        if oversample:
            if RandomOverSampler is None:
                raise RuntimeError("imblearn not installed; set oversample=false")
            ros = RandomOverSampler(random_state=rng)
            Xtr, ytr = ros.fit_resample(Xtr, ytr)

        clf = _get_model(model_name)
        clf.fit(Xtr, ytr)
        yhat = clf.predict(Xte)
        m = _metrics(yte, yhat, labels=labels)
        m["fold"] = fold
        results["fold_metrics"].append(m)

    results["summary"] = _summarise(results["fold_metrics"])
    results["meta"] = meta
    results["config"] = cfg
    return results

def _summarise(fold_metrics):
    keys = [k for k in fold_metrics[0].keys() if k not in ("confusion_matrix", "fold")]
    out = {}
    for k in keys:
        vals = [fm[k] for fm in fold_metrics]
        out[k] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1))}
    return out
