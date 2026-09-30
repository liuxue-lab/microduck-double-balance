"""Read-only head and BAM telemetry; inherited Stage 04 metrics execute once."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from mjlab_microduck.double_balance_stage09_plan import JOINTS, POSTURE_DEG, BUFFER_FRACTION
HARD_DEG = ((-90.,60.),(-90.,90.),(-170.,170.),(-25.,25.))

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def utc():
    return datetime.now(timezone.utc).isoformat()

def qmatrix(q):
    """MuJoCo wxyz quaternion -> active local-to-world rotation; no silent NaN fixes."""
    import numpy as np
    q = np.asarray(q, dtype=np.float64)
    require(q.shape[-1] == 4 and np.isfinite(q).all(), 'Invalid quaternion')
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    require(np.all(np.abs(norm - 1.) < 1e-3), 'Quaternion is not unit length')
    w, x, y, z = np.moveaxis(q / norm, -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                     2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                     2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), axis=-1).reshape(q.shape[:-1]+(3, 3))

def ypr_degrees(matrix):
    """R = Rz(yaw) Ry(pitch) Rx(roll); flag singular samples instead of claiming axes."""
    import numpy as np
    r = np.asarray(matrix)
    cp = np.hypot(r[..., 0, 0], r[..., 1, 0])
    angles = np.stack((np.arctan2(r[..., 1, 0], r[..., 0, 0]),
                       np.arctan2(-r[..., 2, 0], cp),
                       np.arctan2(r[..., 2, 1], r[..., 2, 2])), axis=-1)
    return np.rad2deg(angles), cp > 1e-6

def kinematic_calibration(repo):
    """XML chain at HOME and signed single-joint offsets, independent of simulator state."""
    import numpy as np
    import xml.etree.ElementTree as ET
    root = ET.parse(repo / 'src/mjlab_microduck/robot/microduck/robot_allcollisions.xml').getroot()
    correction = qmatrix([math.sqrt(.5), 0., math.sqrt(.5), 0.])
    home = dict(zip(JOINTS, (.3491, .3491, 0., 0.)))

    def fk(values):
        r = np.eye(3)
        for body_name in ('neck', 'neck_pitch', 'yaw_roll_motion', 'jaw_soft'):
            body = root.find(f".//body[@name='{body_name}']")
            joint = body.find('joint')
            axis = np.fromstring(joint.attrib['axis'], sep=' ')
            angle = values[joint.attrib['name']]
            r = r @ qmatrix(np.fromstring(body.attrib['quat'], sep=' '))
            r = r @ qmatrix(np.r_[math.cos(angle/2), axis*math.sin(angle/2)])
        return r @ correction

    require(np.max(np.abs(fk(home)-np.eye(3))) < 1e-5, 'HOME head frame calibration differs')
    offsets = {}
    for i, name in enumerate(JOINTS):
        limits = np.rad2deg(np.fromstring(root.find(f".//joint[@name='{name}']").attrib['range'], sep=' '))
        require(np.max(np.abs(limits-np.array(HARD_DEG[i]))) < 1e-6, 'XML hard limit differs')
        changed = dict(home)
        changed[name] += math.radians(5.)
        offsets[name] = ypr_degrees(fk(changed))[0].tolist()
    return {'home_joint_degrees': {k: math.degrees(v) for k,v in home.items()},
            'home_head_to_trunk_rotation': fk(home).tolist(),
            'positive_5deg_joint_offset_to_yaw_pitch_roll_degrees': offsets,
            'hard_joint_limits_degrees': dict(zip(JOINTS, HARD_DEG)),
            'aligned_head_frame': 'jaw_soft rotated by the fixed TRAY_QUAT_IN_JAW; +X forward, +Y left, +Z up at HOME',
            'relative_rotation': 'R_trunk_world.T @ R_jaw_world @ R_TRAY_QUAT_IN_JAW',
            'euler_convention': 'active Rz(yaw) Ry(pitch) Rx(roll), right handed, degrees',
            'pitch_sign': 'positive pitch points the forward +X axis downward',
            'raw_joint_angles_are_not_euler_angles': True}

def orientation(trunk, jaw, tray):
    import numpy as np
    rt, rj, rp = qmatrix(trunk), qmatrix(jaw), qmatrix(tray)
    correction = qmatrix([math.sqrt(.5), 0., math.sqrt(.5), 0.])
    rh = rj @ correction
    require(np.max(np.abs(rh-rp)) < 2e-4, 'Head/tray fixed-frame rotation mismatch')
    rel = np.swapaxes(rt, -1, -2) @ rh
    relative, valid = ypr_degrees(rel)
    world, world_valid = ypr_degrees(rh)
    tray_tilt = np.rad2deg(np.arccos(np.clip(rp[...,2,2], -1, 1)))
    total = np.rad2deg(np.arccos(np.clip((np.trace(rel, axis1=-2, axis2=-1)-1)/2, -1, 1)))
    return relative, valid, world, world_valid, tray_tilt, total

def margin(q, limits):
    import numpy as np
    low, high = q-limits[...,0], limits[...,1]-q
    return low, high, np.minimum(low, high)

def angle_statistics(a, valid):
    import numpy as np
    out = {'samples': int(len(a)), 'singular_samples': int((~valid).sum())}
    for j, name in enumerate(('yaw','pitch','roll')):
        values = a[valid,j] if j != 1 else a[:,j]
        if len(values) == 0:
            out[name] = None
            continue
        s, c = np.sin(np.deg2rad(values)).mean(), np.cos(np.deg2rad(values)).mean()
        resultant = math.hypot(float(s),float(c))
        out[name] = {'circular_mean_deg': math.degrees(math.atan2(s,c)) if resultant>1e-8 else None,
                     'circular_resultant': resultant,
                     'median_abs_deg': float(np.median(np.abs(values))),
                     'p95_abs_deg': float(np.quantile(np.abs(values),.95)),
                     'max_abs_deg': float(np.max(np.abs(values))),
                     'fraction_abs_gt_10deg': float((np.abs(values)>10).mean()),
                     'fraction_abs_gt_20deg': float((np.abs(values)>20).mean()),
                     'fraction_abs_gt_45deg': float((np.abs(values)>45).mean())}
    return out

def write_analysis(folder, metadata, data, episodes, metric_trace, metric_columns, active, dt):
    import numpy as np
    folder = Path(folder)
    rows = []
    times = np.arange(1,len(active)+1)*dt
    for episode in episodes:
        i = episode['env_id']
        mask = active[:,i]
        require(mask.sum() == episode['steps'], 'Head samples differ from first-episode step count')
        sample = data['relative_ypr_deg'][mask,i]
        valid = data['euler_valid'][mask,i]
        # Physical 5..10 seconds, not the last five seconds before an early fall.
        tail = mask & (times > 5.0)
        r = {'env_id': i, 'initial_state_id':episode['initial_state_id'],
             'old_success':episode['success'], 'terminated':episode['terminated'],
             'seconds':episode['seconds'], 'terminal_stable_seconds':episode['terminal_stable_seconds'],
             'relative_head_full':angle_statistics(sample, valid),
             'relative_head_5_to_10s':angle_statistics(data['relative_ypr_deg'][tail,i],data['euler_valid'][tail,i]) if tail.any() else None,
             'tray_world_tilt_p95_deg':float(np.quantile(data['tray_tilt_deg'][mask,i],.95)),
             'joints':{}}
        for j,name in enumerate(JOINTS):
            q = np.rad2deg(data['joint_pos_rad'][mask,i,j])
            hard = np.rad2deg(data['hard_min_margin_rad'][mask,i,j])
            normalized = data['hard_margin_fraction_range'][mask,i,j]
            soft = np.rad2deg(data['soft_min_margin_rad'][mask,i,j])
            r['joints'][name] = {'min_deg':float(q.min()),'median_deg':float(np.median(q)),
                                 'max_deg':float(q.max()),'min_hard_margin_deg':float(hard.min()),
                                 'hard_margin_p05_deg':float(np.quantile(hard,.05)),
                                 'fraction_hard_margin_lt_5deg':float((hard<5).mean()),
                                 'fraction_hard_margin_lt_5pct_range':float((normalized<.05).mean()),
                                 'fraction_outside_hard_limits':float((hard<0).mean()),
                                 'min_soft_margin_deg':float(soft.min()),
                                 'fraction_outside_soft_limits':float((soft<0).mean())}
        rows.append(r)
    events = []
    speed_col = metric_columns.index('lower_ball_speed_m_s')
    gate_col = metric_columns.index('pass_lower_ball_speed')
    for episode in episodes:
        i = episode['env_id']
        bad = active[:,i] & (metric_trace[:,i,gate_col] == 0)
        starts = np.flatnonzero(bad & ~np.r_[False,bad[:-1]])
        ends = np.flatnonzero(bad & ~np.r_[bad[1:],False])
        for start,end in zip(starts,ends):
            events.append({'env_id':i,'initial_state_id':episode['initial_state_id'],
                           'start_s':float(times[start]),'end_s':float(times[end]),
                           'sampled_duration_s':float((end-start+1)*dt),
                           'peak_speed_m_s':float(metric_trace[start:end+1,i,speed_col].max()),
                           'head_ypr_deg_at_start':data['relative_ypr_deg'][start,i].tolist(),
                           'joint_speed_deg_s_at_start':np.rad2deg(data['joint_vel_rad_s'][start,i]).tolist(),
                           'analysis_window_start_s':max(0.,float(times[start])-.5),
                           'analysis_window_end_s':min(float(episode['seconds']),float(times[end])+.5)})
    report = {'status':'DIAGNOSTIC_RECORDING_COMPLETE_NOT_A_POSTURE_ACCEPTANCE',
              'metadata':metadata,'control_dt_s':dt,'episodes':rows,'episode_count':len(rows),
              'new_ppo_updates':0, 'task_reward_success_definition_changed':False,
              'angles_thresholds_are_descriptive_bins_not_acceptance':True,
              'soft_margins_definition':'center +/- hard half-range * configured factor; runtime soft limits cross-checked when available',
              'old_successes':sum(e['success'] for e in episodes),
              'pooled_relative_head':angle_statistics(data['relative_ypr_deg'][active],data['euler_valid'][active]),
              'pooled_warning':'time-weighted description; frames and nominal copies are not independent trials',
              'cloud_labels_replaced':False,'head_balance_causality':'NOT_ESTABLISHED',
              'diagnostic_scope':metadata['provenance'].get('protocol','unspecified')}
    np.savez_compressed(folder/'head-trace.npz', **data, metric_values=metric_trace,
                        active_first_episode=active, time_s=times,
                        joint_names=np.array(JOINTS), metric_columns=np.array(metric_columns))
    write(folder/'head-diagnostic.json',report)
    write(folder/'speed-events.json',{'events':events,'causality':'DESCRIPTIVE_ONLY'})
    return report

def make_plots(folder, report, data, metric_values, columns, active, dt):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = report['episodes']
    selections = {0, max(rows,key=lambda r:r['relative_head_full']['yaw']['median_abs_deg'] if r['relative_head_full']['yaw'] else -1)['env_id'],
                  min(rows,key=lambda r:min(j['min_hard_margin_deg'] for j in r['joints'].values()))['env_id']}
    failures = [r for r in rows if not r['old_success']]
    successes = [r for r in rows if r['old_success']]
    if failures: selections.add(failures[0]['env_id'])
    if successes: selections.add(successes[0]['env_id'])
    for i in sorted(selections):
        mask=active[:,i]; t=(np.arange(len(mask))+1)[mask]*dt
        fig,axs=plt.subplots(4,1,figsize=(10,10),sharex=True,layout='constrained')
        ypr=data['relative_ypr_deg'][mask,i].copy()
        ypr[~data['euler_valid'][mask,i],0]=np.nan
        ypr[~data['euler_valid'][mask,i],2]=np.nan
        for j,n in enumerate(('yaw','pitch','roll')): axs[0].plot(t,ypr[:,j],label=n)
        axs[0].set_ylabel('Head / trunk (deg)');axs[0].legend(ncol=3)
        for j,n in enumerate(JOINTS):
            line,=axs[1].plot(t,np.rad2deg(data['joint_pos_rad'][mask,i,j]),label=n)
            for lim in HARD_DEG[j]:axs[1].axhline(lim,color=line.get_color(),ls=':',alpha=.35)
            axs[2].plot(t,np.rad2deg(data['hard_min_margin_rad'][mask,i,j]),label=n)
        axs[1].set_ylabel('Joint angle (deg)');axs[1].legend(ncol=2)
        axs[2].set_ylabel('Hard-limit margin (deg)');axs[2].axhline(0,color='k',ls='--')
        axs[3].plot(t,metric_values[mask,i,columns.index('lower_ball_speed_m_s')],label='lower speed')
        axs[3].axhline(.15,color='r',ls='--',label='unchanged 0.15 m/s limit')
        axs[3].set_ylabel('m/s');axs[3].set_xlabel('Time (s)');axs[3].legend()
        for ax in axs:ax.grid(alpha=.2);ax.set_xlim(0,10)
        ep=next(r for r in rows if r['env_id']==i)
        fig.suptitle(f"env {i} | old_success={ep['old_success']} | terminal stable={ep['terminal_stable_seconds']:.2f}s")
        fig.savefig(Path(folder)/f'head-env-{i:03d}.png',dpi=135);plt.close(fig)
    write(Path(folder)/'plot-selection.json',{'env_ids':sorted(selections),
           'rule':'env0, largest median absolute yaw, smallest hard-limit margin, first failure and success',
           'video_review':'PENDING'})

def observer_class(base, folder, provenance, calibration):
    """Subclass original observer: one original metric call, then read-only data copies."""
    import numpy as np

    def cpu(value):
        return value.detach().cpu().numpy().copy()

    class HeadObserver(base):
        def __init__(self, raw, alive):
            super().__init__(raw,alive)
            self.robot=raw.scene['robot']
            def one(method,name):
                ids,names=method('^'+name+'$')
                require(len(ids)==1, 'Expected one exact '+name)
                return int(ids[0])
            self.jids=[one(self.robot.find_joints,n) for n in JOINTS]
            self.position_action=raw.action_manager.get_term('joint_pos')
            target_ids=self.position_action.target_ids
            if isinstance(target_ids,slice):
                target_ids=list(range(len(self.robot.joint_names)))[target_ids]
            else:target_ids=[int(i) for i in target_ids]
            self.action_columns=[target_ids.index(j) for j in self.jids]
            self.trunk=one(self.robot.find_bodies,'trunk_base')
            self.jaw=one(self.robot.find_bodies,'jaw_soft')
            self.tray=one(self.robot.find_sites,'double_balance_tray_frame')
            require(not any(n.startswith('passive_') for n in self.robot.joint_names),
                    'This diagnostic is for the frozen plain double-balance model')
            self.hard=cpu(self.robot.data.joint_pos_limits[:,self.jids])
            require(np.allclose(np.rad2deg(self.hard),np.array(HARD_DEG)[None],atol=1e-3),
                    'Live model hard limits differ from audited XML')
            factor=float(raw.cfg.scene.entities['robot'].articulation.soft_joint_pos_limit_factor)
            require(0<factor<=1,'Invalid soft limit factor')
            center=self.hard.mean(axis=-1);half=(self.hard[...,1]-self.hard[...,0])*factor/2
            self.soft=np.stack((center-half,center+half),axis=-1)
            actual_soft=getattr(self.robot.data,'soft_joint_pos_limits',None)
            if actual_soft is not None:
                require(np.allclose(cpu(actual_soft[:,self.jids]),self.soft,atol=1e-6),
                        'Runtime soft limits differ from factor-derived limits')
            require(len(self.robot.actuators)==1, 'Expected one BAM group')
            self.bam=self.robot.actuators[0]
            require(hasattr(self.bam,'diagnostics_enabled'), 'BAM telemetry unavailable')
            self.bam.diagnostics_enabled=True
            original_compute=self.bam.compute
            def record_delayed_command(cmd):
                torque=original_compute(cmd)
                self.bam.last_diagnostics['position_target']=cmd.position_target.detach().clone()
                self.bam.last_diagnostics['joint_position']=cmd.pos.detach().clone()
                return torque
            self.bam.compute=record_delayed_command
            self.extra=[]
            self.meta={'provenance':provenance,'calibration':calibration,
                       'joint_names':list(JOINTS),'joint_indices':self.jids,
                       'joint_action_columns':self.action_columns,
                       'actuator_names':list(self.robot.actuator_names),
                       'target_definition':'raw joint_pos action * scale + offset, before downstream clipping/delay',
                       'body_indices':{'trunk_base':self.trunk,'jaw_soft':self.jaw},
                       'tray_site_index':self.tray,'soft_limit_factor':factor,
                       'runtime_soft_limits_cross_checked':actual_soft is not None,
                       'hard_limits_rad':self.hard[0].tolist(),'soft_limits_rad':self.soft[0].tolist(),
                       'sample_phase':'metric cached body poses are pre-final-integration; qpos is post integration; difference 0.002 s',
                       'bam_sample_phase':'last physics substep of each 50 Hz control step; not full substep history',
                       'bam_max_pwm':float(self.bam._bam_model.actuator.max_pwm),
                       'euler_singular_policy':'raw quaternions retained, yaw/roll excluded from summaries'}
            self.record(initial=True)
            # Overlay only on the generated video; physics and policy never read pixels.
            original_render=raw.render
            def render_with_head_values():
                from PIL import Image,ImageDraw,ImageFont
                frame=Image.fromarray(original_render())
                if self.extra:
                    i=int(raw.cfg.viewer.env_idx)
                    latest=self.extra[-1]
                    a=latest['relative_ypr_deg'][i]
                    q=np.rad2deg(latest['joint_pos_rad'][i])
                    m=np.rad2deg(latest['hard_min_margin_rad'][i])
                    lines=[f'Head/trunk Y P R: {a[0]:+.1f} {a[1]:+.1f} {a[2]:+.1f} deg',
                           'Joint angle / hard-limit margin (deg)']
                    lines.extend(f'{name}: {angle:+.1f} / {dist:.1f}' for name,angle,dist in zip(JOINTS,q,m))
                    if not latest['euler_valid'][i]:lines.append('Euler singular: yaw/roll ambiguous')
                    draw=ImageDraw.Draw(frame)
                    draw.rectangle((630,8,frame.width-8,175),fill=(0,0,0))
                    draw.multiline_text((642,15),'\n'.join(lines),fill='white',
                                        font=ImageFont.load_default(size=18),spacing=4)
                return np.asarray(frame)
            raw.render=render_with_head_values

        def record(self,initial=False):
            robot=self.robot
            tq=cpu(robot.data.body_link_quat_w[:,self.trunk])
            hq=cpu(robot.data.body_link_quat_w[:,self.jaw])
            pq=cpu(robot.data.site_quat_w[:,self.tray])
            angles,valid,world,wvalid,tilt,total=orientation(tq,hq,pq)
            q=cpu(robot.data.joint_pos[:,self.jids]);v=cpu(robot.data.joint_vel[:,self.jids])
            action=self.position_action
            target=cpu(action.raw_action*action.scale+action.offset)[:,self.action_columns]
            lo,hi,hard=margin(q,self.hard);_,_,soft=margin(q,self.soft)
            values={'trunk_quat_wxyz':tq,'jaw_quat_wxyz':hq,'tray_quat_wxyz':pq,
                    'relative_ypr_deg':angles,'euler_valid':valid,
                    'head_world_ypr_deg':world,'world_euler_valid':wvalid,
                    'tray_tilt_deg':tilt,'head_relative_total_angle_deg':total,
                    'joint_pos_rad':q,'joint_vel_rad_s':v,
                    'unclipped_joint_target_rad':target,
                    'hard_low_margin_rad':lo,'hard_high_margin_rad':hi,
                    'hard_min_margin_rad':hard,'soft_min_margin_rad':soft,
                    'hard_margin_fraction_range':hard/(self.hard[...,1]-self.hard[...,0]),
                    'policy_action':cpu(self.raw.action_manager.action),
                    'actuator_force':cpu(robot.data.actuator_force)}
            if not initial:
                diagnostics=self.bam.last_diagnostics
                require(diagnostics is not None,'No BAM data after step')
                for key in ('position_target','joint_position','position_error','joint_velocity',
                            'duty_unclipped','duty_current_limited','duty_applied','motor_torque'):
                    value=cpu(diagnostics[key])
                    require(value.shape==(self.raw.num_envs,14),'Unexpected BAM array shape')
                    values['bam_'+key]=value
            from mjlab_microduck.tasks import mdp
            values['stage09_yaw_cost']=cpu(mdp.stage09_head_yaw_cost(self.raw))
            values['stage09_margin_cost']=cpu(mdp.stage09_head_margin_cost(self.raw))
            live=cpu(self.alive).astype(bool)
            require(all(np.isfinite(a[live]).all() for a in values.values()),'Nonfinite live head telemetry')
            if initial:
                np.savez_compressed(Path(folder)/'head-initial.npz',**values)
            else:self.extra.append(values)

        def compute(self):
            # Parent calls raw.metrics_manager's original compute exactly once.
            super().compute()
            from mjlab_microduck.double_balance_stage08_state import assert_zero_assistance
            assert_zero_assistance(self.raw)
            self.record()

        def finish(self,episodes,state_ids,dt):
            trace=super().finish(episodes,state_ids,dt)
            require(len(self.extra)==len(self.rows),'Head/metric sample count mismatch')
            data={k:np.stack([r[k] for r in self.extra]) for k in self.extra[0]}
            active=cpu(trace['active_first_episode']).astype(bool)
            metrics=cpu(trace['values'])
            report=write_analysis(folder,self.meta,data,episodes,metrics,trace['columns'],active,dt)
            write_posture_summary(folder,data,episodes,metrics,trace['columns'],active,dt,self.meta)
            make_plots(folder,report,data,metrics,trace['columns'],active,dt)
            return trace
    return HeadObserver


def write_posture_summary(folder,data,episodes,metrics,columns,active,dt,metadata):
    import numpy as np
    times=np.arange(1,len(active)+1)*dt
    ypr=np.abs(data['relative_ypr_deg'])
    limits=np.asarray(metadata['hard_limits_rad'])
    widths=limits[:,1]-limits[:,0]
    margin=data['hard_min_margin_rad']
    expected_yaw=1.-np.cos(np.deg2rad(data['relative_ypr_deg'][...,0]))
    expected_yaw=np.where(data['euler_valid'],expected_yaw,2.)
    expected_margin=(np.maximum((BUFFER_FRACTION*widths-margin)/(BUFFER_FRACTION*widths),0.)**2).sum(axis=-1)
    yaw_error=float(np.max(np.abs(expected_yaw[active]-data['stage09_yaw_cost'][active])))
    margin_error=float(np.max(np.abs(expected_margin[active]-data['stage09_margin_cost'][active])))
    require(yaw_error<1e-4 and margin_error<1e-4,'Runtime reward differs from independent NumPy telemetry')
    pose_ok=(ypr<=np.array(POSTURE_DEG)).all(axis=-1)&data['euler_valid']
    margin_ok=(margin>=BUFFER_FRACTION*widths).all(axis=-1)
    combined=pose_ok&margin_ok&(metrics[...,columns.index('cached_stable')]>0)
    per_episode=[]
    for e in episodes:
        i=e['env_id'];tail=active[:,i]&(times>5.)
        terminal=0
        for flag in combined[active[:,i],i]:
            terminal=terminal+1 if flag else 0
        head_pass=bool(e['success'] and not e['terminated'] and terminal*dt>=5.-1e-8)
        per_episode.append(dict(env_id=i,initial_state_id=e['initial_state_id'],
            old_success=e['success'],posture_success=head_pass,
            terminal_combined_seconds=terminal*dt,
            tail_yaw_median_deg=float(np.median(ypr[tail,i,0])) if tail.any() else None,
            tail_margin_bad_fraction=float((~margin_ok[tail,i]).mean()) if tail.any() else None))
    tail=active&(times[:,None]>5.)
    command_error=np.abs(data['bam_position_target']-data['bam_joint_position']-data['bam_position_error'])
    command_error_max=float(command_error[active].max())
    require(command_error_max<1e-5,'Recorded BAM command no longer matches the firmware error')
    duty=data['bam_duty_applied'];roll=metadata['actuator_names'].index('head_roll')
    report=dict(bam_command_error_max_rad=command_error_max,reward_numpy_max_errors={'yaw':yaw_error,'margin':margin_error},schema_version=1,metric='stage09-supplemental-posture-v1',old_definition_changed=False,
        thresholds_deg=list(POSTURE_DEG),margin_fraction=BUFFER_FRACTION,
        pose_sample_phase=metadata['sample_phase'],episodes=per_episode,
        successes=sum(e['posture_success'] for e in per_episode),episode_count=len(episodes),
        tail_abs_yaw_median_deg=float(np.median(ypr[...,0][tail])) if tail.any() else None,
        tail_head_margin_bad_fraction=float((~margin_ok[tail]).mean()) if tail.any() else None,
        head_roll_pwm_saturated_fraction=float((np.abs(duty[...,roll][active])>=metadata['bam_max_pwm']-1e-4).mean()),
        bam_phase=metadata['bam_sample_phase'],video_review='PENDING')
    write(Path(folder)/'posture-summary.json',report)
