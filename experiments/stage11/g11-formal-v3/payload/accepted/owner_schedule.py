"""Exact native arithmetic for five adaptive candidates and one baseline."""
from joint_protocol import require
from joint_cases import CASES

def measured(cases):
    require(len(cases)==5,'Five joint branch records required')
    steps=352;trials=4;whole_aborts=0
    for row,(name,mask,mode) in zip(cases,CASES):
        require(row['case']==name and row['mode']==mode and row['logical_mask']==mask,'Joint case schedule changed')
        result=row['physical'];n=result['accepted_resolution']
        require(n in (64,128) and result['mandatory_exact_replay'] and result['replayed'],'Missing original adaptive acceptance/replay')
        used=160 if n==64 else 352;count=3 if n==64 else 4
        require(result['actual_integrations']==used and result['actual_trial_count']==count,'Adaptive trial arithmetic differs')
        steps+=used;trials+=count;whole_aborts+=int('aborted_restore' in row)
    require(1<=whole_aborts<=5,'Initial cancellation required')
    # Six public macro preparations, five branch refreshes, one selected reset.
    expected=dict(integrations=steps,RK4_integrator=steps,position_write=steps,
        GPU_forward=4*steps+trials+8,GPU_solver=4*steps+trials+8,
        sense=trials+15,BVH_refit=trials+15,ray_batch=trials+15,
        state_snapshot=6,state_restore=trials-6,public_clock_write=6,private_clock_write=trials,
        model_option_write=2*trials,endpoint_sensor_read=trials+6,
        CPU_forward=2,CPU_kinematics=1,Data_snapshots=2,Owner_snapshots=2,
        Data_restore=6+whole_aborts,External_Model_restore=6+whole_aborts,Owner_restore=7+whole_aborts,
        Env__reset_idx=1,GPU_reset_data=1,Simulation_reset=1,Scene_reset=1,
        Scene_update=14,Scene_write=6,Action_apply=6,BAM_compute=6,BAM_apply_delay=6,
        Metrics_substep=6,Env_substep_counter=6,sensor_data_reads=252,Startup_BAM_parameter_write=2,
        Action_process_action=6,Observation_compute=6,Observation_noise=6,Observation_delay=6,
        observation_action_reads=7,observation_height_reads=7,Event_apply=11,Command_compute=5,Curriculum_compute=1,
        Recorder_record_pre_reset=1,Recorder_record_post_reset=1)
    for name in ('Action','Observation','Event','Reward','Termination','Metrics','Command','Curriculum'):expected[name+'_reset']=1
    return expected

def verify_counts(counts,cases,caps):
    expected=measured(cases)
    for name,value in expected.items():
        if name=='sensor_data_reads':require(counts[name]==value,'Sensor read count differs');continue
        require(counts.get(name+'_attempted',0)==counts.get(name+'_completed',0)==value,'Native schedule differs: '+name)
    forward=expected['GPU_forward']
    for name,value in dict(module_load=1,allocation=18,capture_begin=1,capture_end=1,launch=forward,kernel_api=33,copy_api=16,graph_destroy=1,exec_destroy=1,cleanup_sync=1).items():
        require(counts.get('ContactGraph_'+name+'_attempted',0)==counts.get('ContactGraph_'+name+'_completed',0)==value,'Sorting graph schedule differs: '+name)
    require(counts.get('Contact_array_assign_attempted',0)==0 and counts['Contact_order_apply_completed']==forward,'Host Contact write or incomplete ordering')
    iterations=counts['Solver_solver_iteration_completed']
    require(0<=iterations<=10*forward,'Solver iteration bound exceeded')
    for name,value in dict(SolverContext_create=forward,Solverinit_context=forward,Solver_update_gradient=forward,
        Solver_update_constraint=forward+iterations,Solver_update_gradient_incremental=iterations,Solver_linesearch=iterations,
        Cost_rows_kernel=forward+iterations,Cost_reduce_kernel=forward+iterations,Cost_override_copy=forward+iterations,
        Norm_kernel=forward+2*iterations,Rigid_metadata_kernel=forward,SolverOrder_module_load=1,SolverOrder_kernel=iterations).items():
        require(counts.get(name+'_attempted',0)==counts.get(name+'_completed',0)==value,'Solver/helper arithmetic differs: '+name)
    for key,value in counts.items():
        if key.endswith('_attempted'):
            require(value==counts.get(key.removesuffix('_attempted')+'_completed',0),'Incomplete native operation: '+key)
            label=key.removesuffix('_attempted')
            if label in caps:require(value<=caps[label],'Declared limit exceeded: '+label)
        require(not key.startswith('blocked_'),'Forbidden entry was attempted')
    return dict(status='COMPLETE_SINGLETON_NATIVE_ACCOUNTING_PASS',expected=expected,solver_iterations=iterations,world_microsteps=expected['integrations'])
