"""赛题 2 输出接口：将预测的进食时间段保存为固定格式 CSV。

只提交预测为进食的时间段，不要把普通非进食时间写入结果。调用格式：

    save_result(userid, start, end, is_eating, output_path='results.csv')

参数含义：

    userid     被试编号，对应用户信息中的 externalid
    start      预测进食开始时间，原始时间轴上的 13 位 Unix 毫秒时间戳
    end        预测进食结束时间，原始时间轴上的 13 位 Unix 毫秒时间戳
    is_eating  1 表示进食并写入；0 表示非进食，不写数据行

start 必须小于 end。时间戳可传整数或整数字符串，不能传秒时间戳、浮点数、
日期字符串、样本序号或相对秒数。is_eating 只接受 0/1 或字符串 '0'/'1'。

示例：

    from output import save_result

    save_result('HNU00001', 1760000000000, 1760000600000,
                1, output_path='results.csv')  # 写入一条进食时段
    save_result('HNU00001', 1760000600000, 1760000700000,
                0, output_path='results.csv')  # 非进食，不写数据行

CSV 列名和顺序固定为 externalid,beforeTime,afterTime，不包含 is_eating 列。
同一文件可以保存多人和多次进食，每次有效调用追加一行，不自动合并或去重。
没有预测出进食时段时，也应提交仅含表头的 CSV。新一轮运行应使用新的空文件，
避免把上一次结果重复追加。validate_result() 只能检查格式，不能判断预测是否正确。

Python 3.9+，仅依赖标准库。
"""

import csv
from numbers import Integral
from pathlib import Path

__all__ = ['save_result', 'validate_result']
HEADER = ('externalid', 'beforeTime', 'afterTime')


def _userid(value):
    if not isinstance(value, str):
        raise ValueError('userid 必须是非空字符串')
    subject = ''.join(value.split()).upper()
    if not subject or any(c in subject for c in (',', '\x00')):
        raise ValueError('userid 无效')
    return subject


def _timestamp(value):
    if isinstance(value, bool):
        raise ValueError('时间戳须为 13 位 Unix 毫秒整数')
    if isinstance(value, Integral):
        text = str(int(value))
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise ValueError('时间戳须为整数或整数字符串，不接受浮点数')
    if len(text) != 13 or not text.isascii() or not text.isdigit() or text.startswith('0'):
        raise ValueError('时间戳须为 13 位 Unix 毫秒整数；不能传秒时间戳或日期字符串')
    return int(text)


def _interval(userid, start, end):
    subject, before, after = _userid(userid), _timestamp(start), _timestamp(end)
    if before >= after:
        raise ValueError('start 必须小于 end')
    return subject, before, after


def _flag(value):
    if isinstance(value, bool):
        raise ValueError('is_eating 只能为 0 或 1')
    if isinstance(value, Integral) and value in (0, 1):
        return int(value)
    if isinstance(value, str) and value in ('0', '1'):
        return int(value)
    raise ValueError('is_eating 只能为 0 或 1')


def validate_result(path):
    """校验表头和每条编号/时间范围，返回 CSV 中的进食时段条数。

    只能校验格式并统计已写入的进食时段，不能判断预测是否正确。
    """
    count = 0
    with Path(path).open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        if next(reader, None) != list(HEADER):
            raise ValueError('CSV 表头必须为 externalid,beforeTime,afterTime')
        for row in reader:
            if len(row) != 3:
                raise ValueError('CSV 每条记录必须恰好有 3 列')
            subject, before, after = _interval(*row)
            if row != [subject, str(before), str(after)]:
                raise ValueError('CSV 编号或时间戳格式不规范')
            count += 1
    return count


def save_result(userid, start, end, is_eating, output_path='results.csv'):
    """将一条预测写入 CSV。返回 True 表示写入，False 表示非进食已跳过。

    同一文件可包含多人、多次进食。每次调用追加，不自动合并或去重。
    返回 True 表示写入进食时段，False 表示 is_eating=0 已跳过。
    即使 is_eating=0 也须提供合法编号和时间；首次调用会创建仅含表头的 CSV。
    """
    flag = _flag(is_eating)
    subject, before, after = _interval(userid, start, end)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('x', encoding='utf-8-sig', newline='') as stream:
            csv.writer(stream).writerow(HEADER)
    except FileExistsError:
        validate_result(path)
    if flag == 0:
        return False
    # 兼容缺少末尾换行的合法 CSV，避免两条记录粘连。
    with path.open('rb') as stream:
        stream.seek(-1, 2)
        needs_newline = stream.read(1) not in (b'\r', b'\n')
    with path.open('a', encoding='utf-8', newline='') as stream:
        if needs_newline:
            stream.write('\r\n')
        csv.writer(stream).writerow((subject, before, after))
    return True
