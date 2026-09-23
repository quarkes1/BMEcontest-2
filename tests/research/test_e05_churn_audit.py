from scripts.research.e05_churn_audit import residual_support_gate


def test_residual_gate_requires_both_mechanisms_and_three_folds():
    assert residual_support_gate(h1=True, h2=True, consistent_folds=3) == "GYRO_RESIDUAL_SUPPORTED"
    assert residual_support_gate(h1=False, h2=True, consistent_folds=5) == "GYRO_RESIDUAL_NOT_SUPPORTED"
    assert residual_support_gate(h1=True, h2=True, consistent_folds=2) == "GYRO_RESIDUAL_NOT_SUPPORTED"
