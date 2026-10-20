"""FN attribution is diagnostic only: distinguish geometry from score/control loss."""
from src.pipeline.event_stack import EventRef, MultiScaleCandidate


def test_missing_candidate_and_low_score_are_not_conflated():
    from scripts.research.official_recovery.fn_audit import audit_candidates
    candidate=MultiScaleCandidate(EventRef('s',0,40),None,None)
    policy=dict(admission=dict(nms_iou=.3,threshold=.2,max_candidates_per_subject=6),threshold=.5)
    rows=audit_candidates([candidate],[.3],['A'],policy,{'s':'A'},[('A',0,100),('A',200,300)],
        set(),{'s':(0,50)})
    assert [r['mechanism'] for r in rows]==['SCORE_BELOW_FINAL_THRESHOLD','NO_MATCHABLE_RAW_CANDIDATE']
    assert rows[0]['raw_reachable'] and not rows[0]['single_file_contains_gt']
    assert rows[0]['max_raw_score']==.3


def test_high_score_capped_candidate_is_control_loss_and_truth_is_counted_once():
    from scripts.research.official_recovery.fn_audit import audit_candidates
    candidates=[MultiScaleCandidate(EventRef('s',a,a+100),None,None) for a in (0,200)]
    policy=dict(admission=dict(nms_iou=.3,threshold=.2,max_candidates_per_subject=1),threshold=.5)
    rows=audit_candidates(candidates,[.9,.8],['A','A'],policy,{'s':'A'},[('A',0,100),('A',200,300)],
        {('A',0,100)},{'s':(0,400)})
    assert [r['mechanism'] for r in rows]==['MATCHED_TP','CONTROL_OR_ONE_TO_ONE_COMPETITION']
    assert rows[0]['legacy_eligible'] and not rows[1]['legacy_eligible']


def test_exact_iou_boundary_is_not_raw_reachable():
    from scripts.research.official_recovery.fn_audit import audit_candidates
    policy=dict(admission=dict(nms_iou=.3,threshold=.2,max_candidates_per_subject=6),threshold=.5)
    rows=audit_candidates([MultiScaleCandidate(EventRef('s',0,25),None,None)],[.9],['A'],policy,
        {'s':'A'},[('A',0,100)],set(),{'s':(0,100)})
    assert rows[0]['mechanism']=='NO_MATCHABLE_RAW_CANDIDATE'
