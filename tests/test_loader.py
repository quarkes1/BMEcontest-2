# -*- coding: utf-8 -*-
import numpy as np
import pytest
from src.data.loader import load_session_tsv, detect_binary

HEADER = "ACC_TIME\tPPG_TIME\tGYRO_TIME\t" + "\t".join(f"PPG{i}" for i in range(1, 45)) + \
         "\tACC_X\tACC_Y\tACC_Z\tGYRO_X\tGYRO_Y\tGYRO_Z"

def make_tsv(tmp_path, lines):
    p = tmp_path / "t.txt"
    p.write_text(HEADER + "\n" + "\n".join("\t".join(map(str, row)) for row in lines), encoding="utf-8")
    return str(p)

def test_load_and_masks(tmp_path):
    # 行1: ACC+GYRO 有效 PPG 全零；行2: 全有效；行3: 全零（末行置零）
    p = make_tsv(tmp_path, [
        [1000, 0, 1000] + [0]*44 + [1.0, 0, 0, 0.1, 0, 0],
        [1095, 1016, 1095] + [5]*44 + [2.0, 0, 0, 0.2, 0, 0],
        [0, 0, 0] + [0]*44 + [0, 0, 0, 0, 0, 0],
    ])
    s = load_session_tsv(p)
    assert s.acc.shape == (3, 3) and s.ppg.shape == (44, 3)
    assert s.imu_valid.tolist() == [True, True, False]
    assert s.ppg_valid.tolist() == [False, True, False]
    assert s.t_acc[2] == -1   # 无效行时间戳置 -1
    assert s.meta["rows"] == 3

def test_detect_binary(tmp_path):
    p = tmp_path / "b.txt"
    p.write_bytes(b"\x00\xff\xf9\x05\x08" * 100)   # 非文本二进制（真实损坏文件形态）
    assert detect_binary(str(p)) is True
    good = tmp_path / "g.txt"
    good.write_text(HEADER + "\n" + "\t".join(["1000", "0", "1000"] + ["0"]*44 + ["1"]*6), encoding="utf-8")
    assert detect_binary(str(good)) is False
