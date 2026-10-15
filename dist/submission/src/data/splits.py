# -*- coding: utf-8 -*-
"""按受试者 GroupKFold 5 折划分，manifest 落盘可复现。"""
import json
from sklearn.model_selection import GroupKFold
import src.config as config
from src.data import manifests

def _fold_path(k):
    return config.CACHE_DIR / "splits" / f"fold{k}.json"

def build_folds():
    index = manifests.load_sensor_index()
    meals = manifests.load_meals()
    # 以有会话数据的受试者为划分单元
    subjects = sorted(index["externalid"].unique())
    groups = index["externalid"].map({s: i for i, s in enumerate(subjects)}).to_numpy()
    kf = GroupKFold(n_splits=5)
    folds = []
    for k, (tr_idx, va_idx) in enumerate(kf.split(index, groups=groups)):
        tr_sessions = set(index.iloc[tr_idx]["session_id"])
        va_sessions = set(index.iloc[va_idx]["session_id"])
        tr_ext = set(index.iloc[tr_idx]["externalid"])
        va_ext = set(index.iloc[va_idx]["externalid"])
        tr_meals = meals[meals["externalid"].isin(tr_ext)]
        va_meals = meals[meals["externalid"].isin(va_ext)]
        fold = {
            "fold": k,
            "train_sessions": sorted(tr_sessions),
            "val_sessions": sorted(va_sessions),
            "train_meals": int(len(tr_meals)),
            "val_meals": int(len(va_meals)),
            "val_scene_balance": {s: int((va_meals["scene"] == s).sum()) for s in config.SCENES},
        }
        folds.append(fold)
        _fold_path(k).parent.mkdir(parents=True, exist_ok=True)
        _fold_path(k).write_text(json.dumps(fold, ensure_ascii=False, indent=1), encoding="utf-8")
    return folds

def load_folds():
    if not _fold_path(0).exists():
        return build_folds()
    return [json.loads(_fold_path(k).read_text(encoding="utf-8")) for k in range(5)]
