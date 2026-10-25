"""Exact two-world Scene reset contracts. NumPy only; no tolerance changes."""
import numpy as np


def require(ok, message):
    if not ok: raise RuntimeError(message)


def exact(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def masks(mask):
    selected = [0] if mask is None else mask
    return selected, [i for i in range(1) if i not in selected]


def owner_reset(before, after, mask, native_time, scalar_before, scalar_after):
    selected, other = masks(mask)
    require(set(before) == set(after), 'Selected owner array roster changed at reset')
    checked = []
    for n, a in before.items():
        b = after[n]
        require(a.shape == b.shape and a.dtype == b.dtype, 'Owner layout changed: '+n)
        if n == 'BAM._delay_buffer._buffer._buffer':
            require(exact(a[:, other], b[:, other]), 'Unselected delay history changed')
            require(np.all(b[:, selected] == 0), 'Selected delay history not zero')
        elif a.ndim and a.shape[0] == 1:
            require(exact(a[other], b[other]), 'Unselected owner array changed: '+n)
            checked.append(n)
    zero = ['BAM._prev_motor_torque', 'BAM._delay_buffer._current_lags',
            'BAM._delay_buffer._step_count', 'BAM._delay_buffer._buffer._num_pushes']
    zero += [n for n in after if n.startswith('entity:') and n.endswith(('.joint_pos_target', '.joint_vel_target', '.joint_effort_target'))]
    zero += [n for n in after if '._air_time_state.' in n and not n.endswith('.last_time')]
    require(len(zero) > 7, 'Entity/contact state coverage missing')
    for n in zero: require(np.all(after[n][selected] == 0), 'Selected reset output not zero: '+n)
    for n in after:
        if '._air_time_state.' in n and n.endswith('.last_time'):
            expected = native_time[selected].reshape((-1,)+(1,)*(after[n].ndim-1))
            require(np.all(after[n][selected] == expected), 'Contact sensor last_time differs: '+n)
    stable = ('vin_tensor', 'vin_drop_gain', 'kp_scale', 'kd_scale', 'default_kp_scale',
              'default_kd_scale', 'friction_scale', 'default_friction_scale')
    for n in stable:
        key='BAM.'+n
        if key in before: require(exact(before[key], after[key]), 'Startup/DR value changed: '+n)
    pointer='BAM._delay_buffer._buffer._pointer'
    require(scalar_before[pointer] == scalar_after[pointer], 'Shared circular pointer reset')
    caches=[n for n in scalar_after if n.startswith('sensor:') and n.endswith('._cache_valid')]
    require(len(caches)==14 and all(scalar_after[n] is False for n in caches), 'Sensor cache reset differs')
    return dict(mask=mask, unselected_owner_fields_exact=len(checked), selected_zero_fields=zero,
                shared_pointer_preserved=True, startup_and_DR_arrays_preserved=True,
                all_14_sensor_caches_invalidated=True)


def append_contract(baseline, after, mask, command, delayed):
    selected, other = masks(mask)
    key='BAM._delay_buffer._buffer._buffer'
    history=after[key]
    for w in selected:
        require(exact(history[:,w], np.broadcast_to(command[w], history[:,w].shape)), 'Selected history not backfilled')
        require(exact(delayed[w], command[w]), 'First post-reset delay output differs')
    pushes=after['BAM._delay_buffer._buffer._num_pushes']
    steps=after['BAM._delay_buffer._step_count']
    require(np.all(pushes[selected]==1) and np.all(steps[selected]==1), 'Selected counters not restarted')
    for n in ('_buffer._num_pushes', '_step_count'):
        key='BAM._delay_buffer.'+n
        require(np.array_equal(after[key][other],baseline[key][other]+1), 'Unselected delay counter differs')
    return dict(first_post_reset_output_exact=True, selected_history_backfill_exact=True)


def post_isolation(control_data, post_data, control_owner, post_owner, control_outputs, post_outputs,
                   control_delayed, post_delayed, mask):
    _, other = masks(mask)
    require(set(control_owner)==set(post_owner), 'Post owner roster changed')
    for n in control_owner:
        a,b=control_owner[n],post_owner[n]
        if n=='BAM._delay_buffer._buffer._buffer':
            require(exact(a[:,other],b[:,other]), 'Unselected post-append history differs')
        elif a.ndim and a.shape[0]==1:
            require(exact(a[other],b[other]), 'Unselected post owner differs: '+n)
    require(exact(control_delayed[other],post_delayed[other]), 'Unselected delayed command differs')
    require(set(control_outputs)==set(post_outputs) and len(post_outputs)==26, 'Sensor output roster differs')
    for n in control_outputs:require(exact(control_outputs[n][other],post_outputs[n][other]), 'Unselected sensor output differs: '+n)
    geo=[n for n in control_data if n.startswith('Data.contact.') and n not in ('Data.contact.efc_address','Data.contact.geomcollisionid')]
    def mapping(d,w):
        count=int(d['Data.nacon'][0]);require(0<=count<=len(d['Data.contact.worldid']),'Bad packed count')
        ids=[i for i in range(count) if int(d['Data.contact.worldid'][i])==w]
        m={tuple(d[k][i].tobytes() for k in geo):i for i in ids}
        require(len(m)==len(ids),'Duplicate canonical contact');return m
    remaps=[]
    for w in other:
        for n,a in control_data.items():
            if n.startswith('Data.contact.') or n=='Data.efc.id' or not a.ndim or a.shape[0]!=1:continue
            require(exact(a[w],post_data[n][w]),'Unselected post Data differs: '+n)
        old,new=mapping(control_data,w),mapping(post_data,w)
        require(old.keys()==new.keys(),'Canonical unselected contact differs')
        recoded=post_data['Data.efc.id'][w].copy()
        for sig,a in old.items():
            b=new[sig];adr=control_data['Data.contact.efc_address'][a]
            require(exact(adr,post_data['Data.contact.efc_address'][b]),'Constraint addresses differ')
            adr=adr[adr>=0]
            require(np.all(control_data['Data.efc.type'][w,adr]==6),'Expected contact constraint type6')
            require(np.all(control_data['Data.efc.id'][w,adr]==a) and np.all(recoded[adr]==b),'Contact ID linkage differs')
            recoded[adr]=a
            if a!=b:remaps.append(dict(world=w,old=a,new=b,rows=adr.tolist()))
        require(exact(control_data['Data.efc.id'][w],recoded),'Unselected constraint ID semantics differ')
    return dict(unselected_delay_outputs_exact=True,unselected_26_sensor_outputs_exact=True,
                unselected_Data_exact_modulo_contact_ID=True,contact_ID_remaps=remaps)
