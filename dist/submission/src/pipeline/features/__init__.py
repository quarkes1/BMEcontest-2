"""训练与推理共用的原始会话特征提取实现。"""

from .macro import MacroFeatureConfig, MacroWindowBatch62, add_time_prior, extract_macro_windows
from .micro import MicroWindowBatch, extract_micro_windows

__all__ = [
    "MacroFeatureConfig",
    "MacroWindowBatch62",
    "MicroWindowBatch",
    "add_time_prior",
    "extract_macro_windows",
    "extract_micro_windows",
]
