# -*- coding: utf-8 -*-
"""赛题 2 输入接口（Python 3.9+，仅依赖标准库）。

数据目录由多个个人文件夹和一个独立用户信息文件组成：

    ACC_test_data/
      userinfobean.txt
      HNU00001/
        collect_data0_....txt
        collect_data1_....txt
      HNU00002/
        ...

参赛程序负责遍历个人目录。DataReader 只读取传入目录直接包含的 TXT，不递归
子目录，也不读取用户信息。read_data() 返回：

    {
      'collect_data0_...': '该 TXT 的完整原始文本',
      'collect_data1_...': '该 TXT 的完整原始文本'
    }

键是文件名去掉 .txt 后的文件 ID；值保留原文件表头、空行、制表符和换行符，
接口不拆列、不转数值、不做信号处理。单个文件较大时，推荐 iter_data()，每次
返回一个完整文件的 (文件ID, 文本)，处理完成后即可释放，避免累计所有文件。

用户信息单独读取：

    from input import DataReader, InfoReader

    person_dir = './ACC_test_data/HNU00001'
    files = DataReader(person_dir).read_data()
    info_reader = InfoReader('./ACC_test_data/userinfobean.txt')
    person_info = info_reader.read_info('HNU00001')

read_info(id) 返回该编号的信息列表；不传编号则返回 {编号: 信息列表}。信息字段
固定为 externalid、height、weight、age、gender，其中身高、体重和年龄为数值。
相同信息会合并，同一编号的冲突信息会全部保留。编号统一转为大写并移除空白。

TXT 使用 UTF-8 编码。空目录返回 {}，空文件返回空字符串，目录不存在抛出
FileNotFoundError。数据读取与用户信息读取互不依赖，导入模块不会读取文件。
"""

import csv
import io
import math
import os
from pathlib import Path

__all__ = ['DataFormatError', 'DataReader', 'InfoReader', 'read_info']
_INFO_FIELDS = ('externalid', 'height', 'weight', 'age', 'gender')


class DataFormatError(ValueError):
    """文件格式错误、字段缺失或数值无法解析。"""


def _subject_id(value):
    subject = ''.join(str(value).split()).upper()
    if not subject:
        raise ValueError('被试编号不能为空')
    return subject


def _finite(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('不允许 NaN 或无穷数值')
    return number


def _read_metadata(path):
    raw = Path(path).read_bytes()  # 仅此小型信息文件一次性加载
    text = None
    for encoding in ('utf-8-sig', 'gb18030'):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            pass
    if text is None:
        raise DataFormatError('%s 无法解码' % Path(path).name)
    reader = csv.DictReader(io.StringIO(text))
    required = set(_INFO_FIELDS)
    if not required.issubset(reader.fieldnames or []):
        raise DataFormatError('%s 缺少必需字段' % Path(path).name)
    result = {}
    for row in reader:
        try:
            if None in row:
                raise ValueError('列数与表头不一致')
            # 仅返回约定字段，避免新增内部字段进入参赛者输入。
            info = {key: row[key].strip() for key in _INFO_FIELDS}
            subject = _subject_id(info['externalid'])
            info['externalid'] = subject
            for key in ('height', 'weight', 'age'):
                info[key] = _finite(info[key])
            records = result.setdefault(subject, [])
            if info not in records:
                records.append(info)
        except (ValueError, AttributeError, TypeError) as exc:
            raise DataFormatError('%s 第 %d 行: %s' % (Path(path).name, reader.line_num, exc)) from exc
    return result


class DataReader:
    """读取一位被试目录下的完整 TXT，不解析内容、不递归子目录。"""

    def __init__(self, dst):
        self.dst = Path(dst)

    def _paths(self):
        if not self.dst.is_dir():
            raise FileNotFoundError('个人目录不存在: %s' % self.dst)
        return sorted(p for p in self.dst.iterdir()
                      if p.is_file() and p.suffix.lower() == '.txt')

    def iter_data(self):
        """按文件名排序，逐个返回 (文件ID, 完整文本)，适合较大的个人目录。"""
        for path in self._paths():
            with path.open('r', encoding='utf-8', newline='') as stream:
                content = stream.read()
            yield path.stem, content

    def read_data(self):
        """返回 {文件名去掉扩展名: 文本内容}；空 TXT 对应空字符串。"""
        result = {}
        for file_id, content in self.iter_data():
            if file_id in result:
                raise DataFormatError('文件 ID 重复: %s' % file_id)
            result[file_id] = content
        return result


class InfoReader:
    """独立读取 userinfobean.txt，不扫描个人数据目录。"""

    def __init__(self, metadata_path=None):
        root = Path(os.environ.get('COMPETITION2_DATA_ROOT')
                    or Path(__file__).parent / 'ACC_test_data')
        self.metadata_path = Path(metadata_path or os.environ.get('COMPETITION2_METADATA')
                                  or root / 'userinfobean.txt')
        self._metadata = _read_metadata(self.metadata_path)

    def read_info(self, subject_id=None):
        """不传编号返回 {编号: 信息列表}；传编号返回该人列表，缺失时为 []。

        返回的是副本，参赛程序修改返回字典不会改变读取器内部信息。
        """
        if subject_id is None:
            return {key: [dict(row) for row in rows] for key, rows in self._metadata.items()}
        return [dict(row) for row in self._metadata.get(_subject_id(subject_id), [])]


def read_info(subject_id=None, metadata_path=None):
    """独立用户信息函数入口；不读取个人数据目录。"""
    return InfoReader(metadata_path).read_info(subject_id)
