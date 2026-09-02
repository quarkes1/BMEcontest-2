# -*- coding: utf-8 -*-
import json
from src.data import splits

def test_build_folds_covers_all_subjects(tmp_path, monkeypatch):
    monkeypatch.setattr(splits.config, "CACHE_DIR", tmp_path)
    folds = splits.build_folds()
    assert len(folds) == 5
    all_subjects = set()
    for f in folds:
        for s in f["train_sessions"]:
            all_subjects.add(s)
        for s in f["val_sessions"]:
            all_subjects.add(s)
        assert set(f["train_sessions"]).isdisjoint(set(f["val_sessions"]))
    assert len(all_subjects) >= 40          # 真实数据 45 名受试者
    assert (tmp_path / "splits" / "fold0.json").exists()
    # 同一 seed 复现
    f2 = splits.build_folds()
    assert f2[0]["val_sessions"] == folds[0]["val_sessions"]
