"""Canonical time-line primitives for raw sensor sessions."""

from .timeline import TimelineSpan, valid_imu_spans, window_starts

__all__ = ["TimelineSpan", "valid_imu_spans", "window_starts"]
