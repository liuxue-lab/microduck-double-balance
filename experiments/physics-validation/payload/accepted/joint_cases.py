"""The only new native cases; fixed reference trials remain archived evidence."""
CASES=(('cancel-control',None,'adaptive'),('control',None,'adaptive'),
       ('reset-0',[0],'adaptive'),('reset-1',[1],'adaptive'),('empty',[],'adaptive'))
PROTOCOL='post-reset-joint-closure-v1'
PHASE_UNITS=10

def check_names(case,world):
    selected=case=='reset-'+str(world)
    names={'check_primary_inputs','check_frozen_reference','check_global_phase_once'}
    names|={'check_Data_reset','check_Manager_reset','check_Scene_reset'} if selected else {'check_immediate_full_noop'}
    if case=='empty' or case=='reset-'+str(1-world):
        names|={'check_unselected_complete_state','check_unselected_complete_stages','check_unselected_full_Model'}
    if case=='control':names|={'check_after_cancellation_complete_state','check_after_cancellation_complete_stages'}
    return names

def check_verdict(row,world):
    actual={k for k in row if k.startswith('check_')}
    if actual!=check_names(row['case'],world):raise RuntimeError('Incomplete joint comparison roster')
    return all(row[k]['passed'] is True for k in actual)
