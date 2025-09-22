io_py = """from typing import Tuple, List
import pandas as pd
import numpy as np
import nibabel as nib
from skimage.measure import label

def load_table(cfg) -> pd.DataFrame:
df = pd.read_csv(cfg["data"]["table_csv"]) # user-provided
# basic sanity
for col in (cfg["data"]["id_col"], cfg["data"]["label_col"]):
if col not in df.columns:
raise ValueError(f"Missing required column: {col}")
return df

def _lesion_component_count(mask_path: str) -> int:
img = nib.load(mask_path)
data = img.get_fdata()
data = (data > 0).astype(np.uint8)
lab = label(data, background=0, connectivity=1)
# subtract 1 if background is labelled
n = lab.max()
return int(n)

def _lesion_voxel_count(mask_path: str) -> float:
img = nib.load(mask_path)
data = img.get_fdata()
return float((data > 0).sum())

def make_features(df: pd.DataFrame, cfg) -> Tuple[np.ndarray, np.ndarray, dict]:
y = df[cfg["data"]["label_col"]].to_numpy()
meta = {"ids": df[cfg["data"]["id_col"]].astype(str).to_list()}

method = cfg[\"features\"][\"method\"]
if method == \"precomputed\":
    cols = cfg[\"features\"].get(\"columns\", [])
    if not cols:
        raise ValueError(\"features.columns must be set when method == 'precomputed'\")
    X = df[cols].to_numpy()
    meta[\"feat_names\"] = cols
    return X, y, meta

elif method == \"lesion_features\":
    mask_col = cfg[\"data\"].get(\"nifti_mask_col\", \"path_mask\")
    if mask_col not in df.columns:
        raise ValueError(f\"Missing mask column '{mask_col}' for lesion_features\")
    feats = []
    names = []
    if cfg[\"features\"][\"lesion\"].get(\"component_count\", True):
        feats.append(df[mask_col].apply(_lesion_component_count).to_numpy())
        names.append(\"lesion_component_count\")
    if cfg[\"features\"][\"lesion\"].get(\"voxel_count\", True):
        feats.append(df[mask_col].apply(_lesion_voxel_count).to_numpy())
        names.append(\"lesion_voxel_count\")
    X = np.vstack(feats).T if feats else np.empty((len(df), 0))
    meta[\"feat_names\"] = names
    return X, y, meta

else:
    raise NotImplementedError(method)

"""
(src / "io.py").write_text(io_py, encoding="utf-8")
