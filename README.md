# ms-lesion-ml (v1.0.0)
Reproducible pipeline for lesion quantification and small-N classification in MRI, adapted from project code by **Yelyzaveta Snihirova**. It provides stratified cross-validation, class-specific metrics (sensitivity/specificity), optional oversampling, and harmonised plotting. All configuration is YAML-based.

> Research use only.

## Key features
- Stratified K-fold CV with fixed seed(s)
- Class-specific metrics: sensitivity, specificity, balanced accuracy
- Optional class balancing via `imblearn`
- Clear separation of I/O, modelling, and plotting
- CLI: `ms-lesion-ml`

## Install
```bash
pip install -e .
```

## Minimal example
1) Copy and edit the template:
```bash
cp config.example.yaml config.yaml
```
2) Train + evaluate:
```bash
ms-lesion-ml fit   --config config.yaml
ms-lesion-ml plot  --config config.yaml
```

## Configuration tips
- `data.table_csv` should have columns: `id`, `label`, and (for lesion features) a `path_mask` column that points to a binary lesion mask NIfTI file per subject.
- For pure tabular runs (no images), you can precompute features and reference them in the CSV; set `features.method: precomputed` and list them in `features.columns`.

## Reproducibility
- Random seeds come from `config.yaml`
- Dependencies pinned in `pyproject.toml`
- Each run saves `outputs/cv_results.json` and `outputs/run_summary.json`

## Licence
MIT. See `LICENSE`.

---

### A note for Key Output
This repository structure (versioned code, CLI, config file, seeded CV, saved metrics/artefacts) is suitable to list as a **software output** in your Kootstra “Key output” section (with a single link to the release/DOI when you mint it).
