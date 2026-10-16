"""Close only constructed Python owners; never call a half-built env.close()."""


def close_environment(env, initialized):
    if env is None:
        return dict(status='NOT_CREATED',closed=[])
    if initialized:
        env.close()
        return dict(status='FULL_ENV_CLOSE_RETURNED',closed=['env.close'])
    closed=[]
    # The pinned ManagerBasedRlEnv.close uses exactly these two attributes.
    # Partial construction may have set neither, one, or both. Preserve the
    # original constructor exception; any real cleanup failure still propagates.
    for name in ('_offline_renderer','recorder_manager'):
        obj=vars(env).get(name)
        if obj is not None:
            obj.close();closed.append(name)
    return dict(status='PARTIAL_CONSTRUCTOR_STOPPED',closed=closed,
                full_environment_constructed=False,resource_release_witness_complete=False,
                remaining_native_allocations='Owned by terminating child process; no reuse or resume')
