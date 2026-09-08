import numpy as np

from scripts import slide_verifier
from src.pipeline import event_stack
from src.pipeline.event_stack import DensityConfig, density_candidates


def sample_windows():
    return {
        "s1": [
            (
                index * 15_000,
                index * 15_000 + 240_000,
                0.8 if 10 <= index <= 19 else 0.1,
            )
            for index in range(40)
        ]
    }


def test_slide_verifier_density_delegates_with_legacy_parity():
    windows = sample_windows()
    legacy = slide_verifier.density_candidates(windows, 0.28838)
    shared = density_candidates(windows, DensityConfig(coverage_fix=False))

    assert slide_verifier.SHARED_DENSITY_CANDIDATES is event_stack.density_candidates
    assert [(candidate[0], candidate[1], candidate[2]) for candidate in legacy] == [
        (candidate.event.sid, candidate.event.start_ms, candidate.event.end_ms)
        for candidate in shared
    ]
    np.testing.assert_allclose(legacy[0][3], shared[0].probabilities)
