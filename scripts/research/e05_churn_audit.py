"""Pure decision rule for the preregistered E05.1 churn audit."""


def residual_support_gate(*, h1: bool, h2: bool, consistent_folds: int) -> str:
    if h1 and h2 and consistent_folds >= 3:
        return "GYRO_RESIDUAL_SUPPORTED"
    return "GYRO_RESIDUAL_NOT_SUPPORTED"
