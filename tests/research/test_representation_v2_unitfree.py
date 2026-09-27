import numpy as np
import torch

from scripts.research.representation_v2.unitfree import normalize_triads, label_starts, rotation_matrices, source_resample, rotate_triads, sample_source_rows, training_priors


def test_unitfree_eliminates_triad_gain_and_fixed_axis_bias():
    x=np.random.default_rng(1).normal(size=(3,6,125))
    mask=np.ones((3,125),bool); mask[:,0:10]=False
    altered=x.copy(); altered[:,:3]=x[:,:3]*1000+np.array([1,2,3])[None,:,None]
    altered[:,3:]=x[:,3:]*.01745+np.array([5,7,9])[None,:,None]
    z,valid=normalize_triads(x,mask); other,valid2=normalize_triads(altered,mask)
    np.testing.assert_allclose(z,other,atol=1e-6)
    assert valid.all() and valid2.all()
    assert np.all(z[:,:,0:10]==0)


def test_constant_or_low_coverage_token_is_invalid_and_finite():
    x=np.zeros((2,6,125)); x[1]=np.random.default_rng(1).normal(size=(6,125))
    mask=np.ones((2,125),bool); mask[1,:26]=False
    z,valid=normalize_triads(x,mask)
    assert not valid.any()
    assert np.isfinite(z).all() and np.all(z==0)


def test_fd_labels_preserve_center_and_drinking_exclusion():
    y=np.zeros(320*5,dtype=int); y[160]=1; y[800]=2
    starts,labels=label_starts(y)
    assert starts.tolist()==[0,1280]
    assert labels.tolist()==[1,0]


def test_source_resampling_does_not_create_motion_from_constant_signal():
    x=np.ones((320,6),np.float32)*np.arange(1,7)[None]
    values=source_resample(x)
    assert values.shape==(6,125)
    # Resample-poly's small periodic ripple is bounded; no zero-pad edge transient.
    np.testing.assert_allclose(values[:,0],np.arange(1,7),atol=.01)
    np.testing.assert_allclose(values[:,-1],np.arange(1,7),atol=.01)


def test_rotations_are_proper_and_share_triad_frame():
    torch.manual_seed(1)
    r=rotation_matrices(8,'cpu')
    torch.testing.assert_close(r@r.transpose(1,2),torch.eye(3).expand(8,-1,-1),atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(torch.linalg.det(r),torch.ones(8),atol=1e-5,rtol=1e-5)
    x=torch.randn(8,3,125)
    y=rotate_triads(torch.cat([x,2*x],dim=1),r)
    torch.testing.assert_close(y[:,3:],2*y[:,:3])


def test_source_sampling_is_independent_of_other_participant_population():
    labels=np.array([0]*300+[1]*50)
    first=sample_source_rows(labels,participant=1)
    sample_source_rows(np.array([0]*1000+[1]*1000),participant=7)
    second=sample_source_rows(labels,participant=1)
    np.testing.assert_array_equal(first,second)
    assert np.sum(labels[first]==0)==256 and np.sum(labels[first]==1)==50


def test_source_priors_exclude_validation_and_distinguish_capped_bank():
    counts=[{'participant':0,'available':{0:90,1:10}}, {'participant':1,'available':{0:1,1:99}}]
    priors=training_priors(counts,[0],np.array([0,0]),np.array([0,1]))
    assert priors=={'sampler':.5,'bank':.5,'eligible_raw':.1}
