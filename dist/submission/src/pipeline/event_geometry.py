"""独立于模型训练的可选整数事件边界校准。"""
from __future__ import annotations

from numbers import Integral

from src.pipeline.event_stack import EventRef

GEOMETRY_SCALES=((1,2),(3,4),(1,1),(5,4),(3,2))
GEOMETRY_SHIFTS=(-60000,0,60000)


def observed_session_bounds(index_bounds,windows):
    """合并元数据边界与实际窗口覆盖范围，不使用预测或真实事件。"""
    result={str(s):tuple(map(int,b)) for s,b in index_bounds.items()}
    if any(a>=b for a,b in result.values()): raise ValueError('invalid index bounds')
    for window in windows:
        if window.sid not in result: raise ValueError('unknown window session')
        if window.start_ms>=window.end_ms: raise ValueError('invalid window bounds')
        a,b=result[window.sid]
        result[window.sid]=(min(a,int(window.start_ms)),max(b,int(window.end_ms)))
    return result


def calibrate_events(events,bounds,scale=(1,1),shift_ms=0):
    if (len(scale)!=2 or any(isinstance(v,bool) or not isinstance(v,Integral) for v in scale)
        or tuple(scale) not in GEOMETRY_SCALES or isinstance(shift_ms,bool)
        or not isinstance(shift_ms,Integral) or shift_ms not in GEOMETRY_SHIFTS):
        raise ValueError('unregistered integer event geometry')
    numerator,denominator=map(int,scale);result=[]
    for event in events:
        if event.sid not in bounds: raise ValueError('unknown event bounds')
        lower,upper=map(int,bounds[event.sid])
        if not lower<=event.start_ms<event.end_ms<=upper: raise ValueError('original event outside session bounds')
        length=max(1,min(upper-lower,(numerator*(event.end_ms-event.start_ms)+denominator//2)//denominator))
        start=(event.start_ms+event.end_ms-length)//2+int(shift_ms)
        start=max(lower,min(upper-length,start))
        result.append(EventRef(event.sid,int(start),int(start+length)))
    return result
