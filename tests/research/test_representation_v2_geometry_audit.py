from dataclasses import asdict
from scripts.research.representation_v2.geometry_audit import match_events, compare_geometry, clip_support
from src.pipeline.event_stack import EventRef, CandidateEvent, MultiScaleCandidate


def test_exact_iou_boundary_and_session_identity():
    truth=[EventRef('a',0,100)]
    prediction=[EventRef('a',0,25),EventRef('b',0,100)]
    assert len(match_events(prediction,truth,strict=False))==1
    assert match_events(prediction,truth,strict=True)==[]


def test_same_counts_can_hide_changed_matched_truths():
    truths=[EventRef('s',0,100),EventRef('s',200,300)]
    result=compare_geometry([EventRef('s',0,100)],[EventRef('s',200,300)],truths)
    assert result['baseline']['tp']==result['clipped']['tp']==1
    assert result['matched_gt_gained']==1 and result['matched_gt_lost']==1
    assert result['common_matched_gt']==0


def test_clipping_is_nonmutating_and_single_source_unchanged():
    macro=CandidateEvent(EventRef('s',0,100),(1.,),1.,0,0,0,0)
    micro=CandidateEvent(EventRef('s',20,120),(1.,),1.,0,0,0,0)
    candidate=MultiScaleCandidate(macro.event,macro,micro)
    before=asdict(candidate)
    assert clip_support(candidate)==EventRef('s',20,100)
    assert asdict(candidate)==before
    assert clip_support(MultiScaleCandidate(macro.event,macro,None))==macro.event
