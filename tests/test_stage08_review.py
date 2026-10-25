"""Reject reward-only improvement, incompatible data and unearned extensions."""
import json
import pytest
from mjlab_microduck.double_balance_stage08_review import improvement_gate, reference, validate_decision
from mjlab_microduck.double_balance_stage08_plan import TRAIN_SEEDS


def report(k=0,seed=TRAIN_SEEDS[0]):
    return {'status':'PASS','protocol':'dev','protocol_version':'v1',
            'initial_states':{'sha256':'same-states'},'profile':'E','seed':seed,
            'experiment_completed_updates':k,'campaign_id':'campaign','checkpoint_sha256':'model',
            'success_fraction':0.,'termination_fraction':0.,'mean_survival_seconds':10.,
            'median_terminal_stable_seconds':.1,'median_longest_stable_seconds':2.,
            'mean_lower_speed_violation_fraction':.1}


def pair(seed=TRAIN_SEEDS[0],ks=(250,500)):
    return [{**report(k,seed),'median_terminal_stable_seconds':.2,
             'median_longest_stable_seconds':3.,'mean_lower_speed_violation_fraction':.07} for k in ks]


def test_reward_only_change_is_not_an_extension_reason():
    baseline=report()
    reports=[{**report(k),'training_reward':10000} for k in (250,500)]
    assert not improvement_gate(reports,baseline)['passed']
    assert improvement_gate(pair(),baseline)['passed']
    reports[1]['success_fraction']=.1
    assert improvement_gate(reports,baseline)['passed']
    reports[1]['termination_fraction']=.5
    assert not improvement_gate(reports,baseline)['passed']


def test_non_adjacent_or_different_initial_states_rejected():
    reports=pair(ks=(250,1000))
    with pytest.raises(ValueError,match='adjacent'):
        improvement_gate(reports,report())
    reports=pair(); reports[1]['initial_states']={'sha256':'other-states'}
    with pytest.raises(ValueError,match='comparable'):
        improvement_gate(reports,report())


def test_extension_needs_all_three_seeds_and_unmodified_evidence(tmp_path):
    def write(name,value):
        path=tmp_path/name; path.write_text(json.dumps(value)); return reference(path)
    baseline=write('base.json',report())
    evidence=[write(f'{seed}-{r["experiment_completed_updates"]}.json',r)
              for seed in TRAIN_SEEDS for r in pair(seed,(1250,1500))]
    decision={'schema_version':1,'campaign_id':'campaign','kind':'extension',
              'selected_profiles':['E'],'baseline_report':baseline,'candidate_reports':evidence}
    validate_decision(decision,profile='E',target=4000,campaign_id='campaign')
    with pytest.raises(ValueError,match='three seeds'):
        validate_decision({**decision,'candidate_reports':evidence[:2]},profile='E',target=4000,campaign_id='campaign')
    (tmp_path/'base.json').write_text('{}')
    with pytest.raises(ValueError,match='changed'):
        validate_decision(decision,profile='E',target=4000,campaign_id='campaign')
