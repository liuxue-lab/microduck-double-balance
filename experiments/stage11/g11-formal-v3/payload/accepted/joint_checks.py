"""Original physical criteria plus explicit replicated-global-phase witnesses."""
import copy
import numpy as np
from lifecycle_support import require
from ground_scalar_contract import assess,criteria
from convergence_contract import compare
from transaction_contract import exact
from owner_checks import global_timer_seal

def frozen_reference(accepted,post,reference,world):
    require(len(accepted['worlds'])==1 and accepted['resolution'] in (64,128),'Complete accepted singleton64/128 required')
    a=accepted['worlds'][0];b=copy.deepcopy(reference['worlds'][world])
    for n in ('qpos','qvel'):b['endpoint'][n]=np.asarray(b['endpoint'][n],np.float32)
    current=float(post['data']['Data.time'][0]);historical=reference['native_times'][world]
    require(current==historical,'Current and frozen reference clocks differ')
    local=assess(a,b)
    width=max((r['width_s'] for r in a['brackets']),default=0.)
    own_width=width<=criteria()['accepted_event_bracket_width_s']
    def point(x,t):return dict(tick=1,native_time_s=t,qpos=x['endpoint']['qpos'],qvel=x['endpoint']['qvel'])
    physical=compare([point(a,current)],[point(b,historical)],a['integral'],b['integral'],expected_macros=1,max_resolution=128)
    return dict(passed=bool(local['accepted'] and own_width and physical['passed']),logical_world=world,
        current_resolution=accepted['resolution'],fixed_reference_resolution=128,source_member=reference['source_member'],source_sha256=reference['source_sha256'],
        local_state_event=local,current_bracket_width_s=width,current_bracket_passed=own_width,state_contact=physical,
        original_criteria_unchanged=True,reference_native_calls=0,complete_opaque_history_equivalence_claimed=False)

def global_phase_once(base,immediate,post):
    require(global_timer_seal(base)==global_timer_seal(immediate),'Local reset changed global timer or stream')
    name='EventManager._interval_term_time_left.2';a=base['owners'][name];b=post['owners'][name]
    expected=a-np.float32(.002)
    require(expected.shape==(1,) and bool((expected>=1e-6).all()),'Declared short macro unexpectedly crosses global expiry')
    require(exact(b,expected),'Global phase did not advance exactly once')
    key='RNGService.states.event_global_2/0'
    require(exact(base['owners'][key],post['owners'][key]),'Unexpected global RNG draw')
    require(exact(base['owners']['env.events'][3],post['owners']['env.events'][3]),'Unexpected global event firing')
    for kind in ('draw_counts','seeds'):
        key='RNGService.'+kind+'.event_global_2/0'
        require(base['scalars'][key]==post['scalars'][key],'Global stream metadata changed')
    return dict(phase_before=a.tolist(),phase_after=b.tolist(),dt=.002,advances=1,reset_preserved=True,
                global_expiry_triggered=False,global_rng_unchanged=True)
