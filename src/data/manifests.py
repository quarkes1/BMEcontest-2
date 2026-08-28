# -*- coding: utf-8 -*-
"""会话清单 / 标签 / 用户表加载与归一化。"""
import csv
import pandas as pd
import src.config as config

MEALS_CSV = config.DATA_DIR / "t_zsstnnrj_mealinfo_puadqog70826_1857.csv"
USERS_CSV = config.DATA_DIR / "t_zsstnnrj_userinfobean_5d4l0nmp0826_1857.csv"
INDEX_CSV = config.DATA_DIR / "t_zsstnnrj_sensororiginaldata_system0826_1857.csv"
BLACKLIST_FILE = config.OUTPUT_DIR / "data_quality_blacklist.txt"   # 数据校验产出，无则空

def normalize_hand(value: str) -> str:
    """'左手佩戴' -> '左手'; '右手' -> '右手'"""
    return str(value).replace("佩戴", "").strip()

def scene_of(wear_hand: str, dietary_hand: str) -> str:
    """wear==dietary -> dominant（惯用手场景），否则 nondominant"""
    return "dominant" if normalize_hand(wear_hand) == normalize_hand(dietary_hand) else "nondominant"

def load_blacklist() -> set:
    if BLACKLIST_FILE.exists():
        return {line.strip() for line in BLACKLIST_FILE.read_text(encoding="utf-8").splitlines() if line.strip()}
    return set()

def load_meals() -> pd.DataFrame:
    """用餐标签：剔除 test 用户与无效（after<=before）记录，附加 scene/duration 列。"""
    with open(MEALS_CSV, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        ext = str(r["externalid"]).strip()
        if not ext or ext.lower() == "test":
            continue
        before, after = int(float(r["beforeTime"])), int(float(r["afterTime"]))
        if after <= before:
            continue
        out.append({
            "meal_id": f"{ext}_{before}",
            "externalid": ext,
            "scene": scene_of(r["wearHand"], r["dietaryHand"]),
            "dietary_type": str(r["dietaryType"]).strip(),
            "before_ms": before, "after_ms": after,
            "tableware": str(r["tablewareType"]).strip(),
            "duration_min": round((after - before) / 60000.0, 2),
        })
    return pd.DataFrame(out)

def load_users() -> pd.DataFrame:
    df = pd.read_csv(USERS_CSV, encoding="utf-8", dtype={"externalid": str})
    df = df[df["externalid"].str.lower() != "test"].drop_duplicates(subset="externalid")
    return df

def load_meal_meta():
    """每受试者的用餐区间元数据（场景/餐具/膳食类型），供 W2 窗口缓存脚本共用。
    返回 (meta, tableware_classes)：meta[ext] = [{before,after,scene,tableware,dietary}, ...]"""
    meals = load_meals()
    tw_classes = sorted(meals["tableware"].unique().tolist())
    tw_idx = {v: i for i, v in enumerate(tw_classes)}
    meta = {}
    for _, r in meals.iterrows():
        meta.setdefault(r["externalid"], []).append({
            "before": int(r["before_ms"]), "after": int(r["after_ms"]),
            "scene": r["scene"], "tableware": tw_idx[r["tableware"]],
            "dietary": r["dietary_type"]})
    return meta, tw_classes

def load_sensor_index() -> pd.DataFrame:
    """传感器索引：session_id=zip 名 stem；剔除空/NaN、test 用户与黑名单会话。"""
    df = pd.read_csv(INDEX_CSV, encoding="utf-8", dtype={"externalid": str})
    df["session_id"] = df["sensorData"].map(lambda p: p.split("/")[-1].removesuffix(".zip"))
    df = df[df["externalid"].notna() & (df["externalid"].str.strip() != "")
            & (df["externalid"].str.lower() != "test")]
    blacklist = load_blacklist()
    df = df[~df["session_id"].isin(blacklist)]
    return df
