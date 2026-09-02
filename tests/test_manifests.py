# -*- coding: utf-8 -*-
import pytest
import pandas as pd
from src.data.manifests import scene_of, normalize_hand, load_meals

def test_scene_of():
    assert scene_of("右手佩戴", "右手") == "dominant"
    assert scene_of("左手佩戴", "右手") == "nondominant"
    assert scene_of("左手佩戴", "左手") == "dominant"

def test_normalize_hand():
    assert normalize_hand("左手佩戴") == "左手"
    assert normalize_hand("右手") == "右手"

def test_load_meals_excludes_test_and_invalid(tmp_path, monkeypatch):
    csv = tmp_path / "meals.csv"
    csv.write_text(
        "externalid,dietaryType,beforeTime,afterTime,drinkingVolume,satiety,foodName,"
        "tablewareType,dietaryHand,wearHand\n"
        "HNU21001,晚餐,1784457642000,1784458542000,300mL,刚刚好,米饭,筷子,右手,左手佩戴\n"
        "test,晚餐,1784457642000,1784457642000,300mL,刚刚好,米饭,筷子,右手,左手佩戴\n"
        "HNU21001,午餐,1784459642000,1784459642000,300mL,刚刚好,米饭,筷子,右手,左手佩戴\n",
        encoding="utf-8")
    monkeypatch.setattr("src.data.manifests.MEALS_CSV", csv)
    df = load_meals()
    assert len(df) == 1
    assert df.iloc[0]["scene"] == "nondominant"
    assert df.iloc[0]["duration_min"] == pytest.approx(15.0)
