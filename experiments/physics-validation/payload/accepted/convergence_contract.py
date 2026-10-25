"""Predeclared, bounded numerical agreement; not physical material calibration."""
from pathlib import Path
import json
import math
import numpy as np
from lifecycle_support import require

PAIRS = {'upper_head': (49,84), 'lower_floor': (0,82),
         'lower_left_foot': (29,82), 'lower_right_foot': (79,82), 'upper_floor': (0,84), 'left_foot_floor': (0,29), 'right_foot_floor': (0,79)}

def criteria():
    return json.loads(Path(__file__).with_name('acceptance-criteria.json').read_text())

class ContactIntegral:
    def __init__(self):
        self.stages = 0
        self.duration = 0.
        self.pairs = {name: dict(normal_impulse_Ns=0., loaded_time_s=0., peak_overlap_m=0.) for name in PAIRS}

    def observe(self, row, stage, dt):
        require(stage in (0,1,2,3) and math.isfinite(dt) and dt>0, 'Invalid quadrature')
        require(stage == self.stages % 4, 'RK quadrature stage order differs')
        weight = dt * (1,2,2,1)[stage] / 6
        grouped = {name: [] for name in PAIRS}
        for contact in row['contact_force_rows']:
            pair = tuple(sorted(map(int,contact['geoms'])))
            names = [name for name,p in PAIRS.items() if p == pair]
            for name in names:
                forces = contact['raw_solver_forces']
                dim=3 if name in ('left_foot_floor','right_foot_floor') else 4
                require(len(forces) in (0,2*(dim-1)) and all(math.isfinite(float(f)) for f in forces), 'Expected scoped pyramidal contact rows')
                grouped[name].append((float(contact['distance_m']), math.fsum(forces)))
        for name, contacts in grouped.items():
            result = self.pairs[name]
            result['normal_impulse_Ns'] += weight * math.fsum(force for _,force in contacts)
            result['loaded_time_s'] += weight * any(gap<=0 and force>0 for gap,force in contacts)
            result['peak_overlap_m'] = max(result['peak_overlap_m'], max((-gap for gap,_ in contacts), default=0.))
        self.stages += 1
        self.duration += weight

    def result(self):
        return dict(stages=self.stages, quadrature_duration_s=self.duration, pairs=self.pairs)

def quaternion_angle(a, b):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    require(np.linalg.norm(a)>0 and np.linalg.norm(b)>0, 'Zero quaternion')
    a=a/np.linalg.norm(a);b=b/np.linalg.norm(b)
    if np.dot(a,b)<0:b=-b
    return float(4*np.arctan2(np.linalg.norm(a-b),np.linalg.norm(a+b)))

def compare(coarse, fine, coarse_integral, fine_integral,expected_macros=50,max_resolution=128):
    """All 50 macro boundaries, full physical coordinates, seven contact pairs."""
    rule=criteria()
    require(len(coarse)==len(fine)==expected_macros, 'Agreement requires every declared macro endpoint')
    maxima={name:dict(value=0.,tick=None,component=None) for name in rule['state_limits']}
    def record(name,value,tick,component):
        require(math.isfinite(value), 'Nonfinite agreement metric')
        if value>=maxima[name]['value']:maxima[name]=dict(value=float(value),tick=tick,component=component)
    for tick,(a,b) in enumerate(zip(coarse,fine),1):
        require(a['tick']==b['tick']==tick and a['native_time_s']==b['native_time_s'], 'Unaligned endpoint clocks')
        aq,bq=np.asarray(a['qpos'],dtype=float),np.asarray(b['qpos'],dtype=float)
        av,bv=np.asarray(a['qvel'],dtype=float),np.asarray(b['qvel'],dtype=float)
        require(aq.shape==bq.shape==(35,) and av.shape==bv.shape==(32,), 'Physical state dimensions differ')
        require(all(np.isfinite(x).all() for x in (aq,bq,av,bv)), 'Nonfinite state')
        for body,qa,va in (('robot',0,0),('lower',21,20),('upper',28,26)):
            record('free_translation_m',np.linalg.norm(aq[qa:qa+3]-bq[qa:qa+3]),tick,body)
            record('free_rotation_rad',quaternion_angle(aq[qa+3:qa+7],bq[qa+3:qa+7]),tick,body)
            record('free_linear_velocity_m_s',np.linalg.norm(av[va:va+3]-bv[va:va+3]),tick,body)
            record('angular_velocity_rad_s',np.linalg.norm(av[va+3:va+6]-bv[va+3:va+6]),tick,body)
        for q in range(7,21):
            record('hinge_position_rad',abs(aq[q]-bq[q]),tick,'qpos:'+str(q))
            record('angular_velocity_rad_s',abs(av[q-1]-bv[q-1]),tick,'qvel:'+str(q-1))
    checks=[]
    for name,witness in maxima.items():
        limit=rule['state_limits'][name]
        checks.append(dict(metric=name,**witness,limit=limit,passed=witness['value']<=limit))
    for label,x in (('coarse',coarse_integral),('fine',fine_integral)):
        require(expected_macros*32*4<=x['stages']<=expected_macros*max_resolution*4 and x['stages']%128==0 and abs(x['quadrature_duration_s']-expected_macros*.002)<1e-8, 'Incomplete contact quadrature: '+label)
    for name in PAIRS:
        a,b=coarse_integral['pairs'][name],fine_integral['pairs'][name]
        for metric in ('peak_overlap_m','normal_impulse_Ns','loaded_time_s'):
            limit=rule['contact_limits'][metric]
            threshold=limit if isinstance(limit,(int,float)) else limit['absolute']+limit['relative']*abs(b[metric])
            error=abs(a[metric]-b[metric])
            require(math.isfinite(error), 'Nonfinite contact metric')
            checks.append(dict(metric=metric,pair=name,coarse=a[metric],fine=b[metric],value=error,limit=threshold,passed=error<=threshold))
    return dict(passed=all(c['passed'] for c in checks),checks=checks,failed_metrics=[c for c in checks if not c['passed']],
                compared_endpoints=expected_macros,interpretation='Bounded numerical agreement only; no material calibration or RK order claim')
