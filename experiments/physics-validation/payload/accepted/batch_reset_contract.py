"""Read-only NumPy checks for native two-world reset, without simulator imports.

Packed contacts are global. Only after forward are nacon/worldid meaningful as
a rebuilt contact set. No tolerance is introduced by this preflight.
"""
import numpy as np

PRIMARY = ('time', 'qpos', 'qvel', 'act', 'ctrl', 'qfrc_applied',
           'xfrc_applied', 'eq_active', 'mocap_pos', 'mocap_quat')
ZERO = ('solver_niter', 'ne', 'nf', 'nl', 'nefc', 'time', 'energy', 'qvel',
        'qacc_warmstart', 'qfrc_applied', 'qacc', 'ctrl', 'act', 'act_dot',
        'sensordata', 'qM', 'xfrc_applied')
CONTACT = ('dist', 'pos', 'frame', 'includemargin', 'friction', 'solref',
           'solreffriction', 'solimp', 'dim', 'geom', 'flex', 'vert',
           'efc_address', 'worldid', 'type', 'geomcollisionid')
GEOMETRY = tuple(x for x in CONTACT if x not in ('efc_address', 'geomcollisionid'))


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def exact(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def contact_rows(raw, world):
    n = int(raw['Data.nacon'][0])
    ids = raw['Data.contact.worldid']
    require(0 <= n <= len(ids), 'Contact count outside allocated storage')
    require(np.isin(ids[:n], [0]).all(), 'Invalid live contact world ID')
    rows = [tuple(raw['Data.contact.'+f][i].tobytes() for f in GEOMETRY)
            for i in range(n) if int(ids[i]) == world]
    return sorted(rows)


def check_reset(before, after, mask, defaults):
    selected = [0] if mask is None else mask
    other = [i for i in range(1) if i not in selected]
    require(set(before) == set(after), 'Data schema changed')
    checked = []
    # All non-contact arrays with a leading world dimension: no shared counters
    # are inferred to be per-world just because their sizes happen to match.
    for name, a in before.items():
        if not name.startswith('Data.contact.') and name != 'Data.nacon' and a.ndim and a.shape[0] == 1:
            require(exact(a[other], after[name][other]), 'Unselected reset bytes changed: '+name)
            checked.append(name)
    for name in ZERO:
        a = after['Data.'+name][selected]
        require(np.all(a == 0), 'Selected reset output not zero: '+name)
    for world in selected:
        qpos0 = defaults['qpos0'][world % defaults['qpos0'].shape[0]]
        require(exact(after['Data.qpos'][world], qpos0), 'qpos0 reset differs')
        require(exact(after['Data.eq_active'][world], defaults['eq_active0']), 'eq_active0 reset differs')
        for field in ('mocap_pos', 'mocap_quat'):
            require(exact(after['Data.'+field][world], defaults[field][world]), 'Mocap defaults differ')
    n = int(before['Data.nacon'][0])
    expected_n = 0 if 0 in selected else n
    require(int(after['Data.nacon'][0]) == expected_n, 'Shared nacon reset differs from pinned source')
    worlds = before['Data.contact.worldid'][:n]
    clear = np.isin(worlds, selected)
    for field in CONTACT:
        name = 'Data.contact.'+field
        a, b = before[name], after[name]
        require(exact(a[:n][~clear], b[:n][~clear]), 'Unselected packed contact changed: '+field)
        require(exact(a[n:], b[n:]), 'Inactive packed storage changed: '+field)
        value = -1 if field == 'efc_address' else 0
        require(np.all(b[:n][clear] == value), 'Selected packed contact not cleared: '+field)
    if not selected:
        require(all(exact(a, after[n]) for n, a in before.items()), 'Empty reset changed Data')
    return dict(mask=mask, selected=selected, unselected=other, batched_fields_checked=checked,
                pre_nacon=n, immediate_nacon=expected_n,
                selected_contact_rows=int(clear.sum()), unselected_contact_rows=int((~clear).sum()),
                immediate_contact_set_is_not_a_forward_result=True)


def check_forward(control, after, selected):
    other = [i for i in range(1) if i not in selected]
    for name in PRIMARY:
        require(exact(control['Data.'+name][other], after['Data.'+name][other]),
                'Unselected primary state differs after forward: '+name)
    contacts = {}
    for world in other:
        a, b = contact_rows(control, world), contact_rows(after, world)
        require(a == b, 'Unselected canonical contact geometry differs after forward')
        contacts[str(world)] = len(a)
    derived = {}
    for name, a in control.items():
        if name.startswith('Data.contact.') or name == 'Data.nacon' or not a.ndim or a.shape[0] != 1:
            continue
        b = after[name]
        x, y = a[other], b[other]
        equal = exact(x, y)
        row = {'exact': equal, 'elements': int(x.size)}
        if x.dtype.kind in 'iuf' and x.size:
            finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
            row['finite'] = finite
            if finite:
                row['max_abs_difference'] = float(np.max(np.abs(x.astype('float64')-y.astype('float64'))))
        derived[name] = row
    return dict(unselected_contact_counts=contacts, primary_exact=True,
                canonical_contact_geometry_exact=True, derived_field_observations=derived,
                derived_solver_acceptance=False, G09='OPEN')


def exercise(masks, restore, forward, reset, record, check_immediate, check_post, milestone):
    """Exactly two restores/two forwards/one reset per mask; no retries."""
    rows = []
    for i, mask in enumerate(masks):
        prefix = 'case-%02d' % i
        restore()
        forward()
        control = record(prefix+'-control')
        restore()
        reset(mask)
        immediate = record(prefix+'-reset')
        a = check_immediate(immediate, mask)
        forward()
        post = record(prefix+'-forward')
        b = check_post(control, post, a['selected'])
        row = dict(index=i, mask=mask, immediate=a, post_forward=b)
        rows.append(row)
        milestone(row)
    return rows
