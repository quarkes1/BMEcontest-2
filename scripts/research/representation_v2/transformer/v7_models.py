"""Authorized train-only loading and persisted candidate producer inference."""
import numpy as np

from .. import event_probe as ep
from ..crossfit import CandidateLedger, _slice_batch


class SubsetSource(ep.runner.FilesystemDataSource):
    """Filter shared feature tables before raw eligibility/session processing."""
    def __init__(self,root,allowed_sids):
        super().__init__(root)
        self.allowed_sids=set(allowed_sids)
        if not self.allowed_sids: raise ValueError('empty allowed session set')

    def _load_batch(self,path):
        return _slice_batch(super()._load_batch(path),self.allowed_sids)

    def _load_micro_batch(self,path,expected_metadata):
        batch,seconds=super()._load_micro_batch(path,expected_metadata)
        return _slice_batch(batch,self.allowed_sids),seconds


def fit_producers_and_base(data,config,ledger):
    models={'train':sorted(set(data.subject_by_session.values())),
            'micro_threshold':ledger.micro_threshold}
    for name,batch,keep,model in (
        ('macro',data.window_train,np.asarray(data.window_train.labels)>=0,ep.runner._window_estimator(config.seed)),
        ('micro',data.micro_window_train,ep.runner._micro_training_keep(data.micro_window_train,data.train_truths,
                    config.micro_positive_middle_fraction),ep.runner._micro_window_estimator(config.seed+10))):
        x=ep.runner._with_time_prior(batch.features,batch.windows) if name=='macro' else batch.features
        y=np.asarray(batch.labels[keep],np.int8)
        ep.runner._validate_binary(y,'V7 final producer')
        model.fit(x[keep],y); models[name]=model
    models['base']=[]
    for model in (ep.runner._verifier_estimator(config.seed+20,.1),ep.runner._verifier_lgbm_estimator(config.seed+21)):
        model.fit(ledger.features,ledger.labels); models['base'].append(model)
    return models


def base_probability(models,features):
    return np.mean([ep.runner._positive_probability(m,features) for m in models['base']],axis=0)


def score_ledger(data,config,models):
    """No fitting or truth/label use; final producer models were sealed earlier."""
    score=sorted(set(data.subject_by_session.values()))
    if set(score)&set(models['train']): raise ValueError('refit/score subject overlap')
    macro=ep.runner._windows_by_session(data.candidate_train.windows,
        ep.runner._positive_probability(models['macro'],ep.runner._with_time_prior(
            data.candidate_train.features,data.candidate_train.windows)))
    micro=ep.runner._windows_by_session(data.micro_candidate_train.windows,
        ep.runner._positive_probability(models['micro'],data.micro_candidate_train.features))
    candidates=tuple(ep.runner.union_candidates(ep.runner.density_candidates(macro,config.density),
        ep.runner.micro_candidates(micro,models['micro_threshold'],config.micro_candidate)))
    features=ep.runner.multiscale_verifier_features(candidates,macro,micro,context_features_version='v1',
        session_bounds_by_sid=ep.runner._context_session_bounds(data,candidates,
            data.candidate_train.windows,data.micro_candidate_train.windows))
    if features.shape!=(len(candidates),116): raise ValueError('V7 feature contract differs')
    groups=ep.runner._groups_for([c.event for c in candidates],data.subject_by_session)
    return CandidateLedger(candidates,features,np.zeros(len(candidates),np.int8),groups,
        models['micro_threshold'],tuple(models['train']),tuple(score))
