"""Use the pinned real environment class without constructing a physical env."""
from integration.environment_cleanup import close_environment


def verify(env_class):
    rows=[]
    class Owner:
        def __init__(self, fail=False):
            self.calls=0;self.fail=fail
        def close(self):
            self.calls+=1
            if self.fail:raise ValueError('cleanup-fixture-failure')
    assert close_environment(None,False)['status']=='NOT_CREATED'
    rows.append('no-environment')
    for names in ((),('_offline_renderer',),('recorder_manager',),('_offline_renderer','recorder_manager')):
        env=env_class.__new__(env_class)
        owners={name:Owner() for name in names}
        vars(env).update(owners)
        result=close_environment(env,False)
        assert result['closed']==list(names) and not result['resource_release_witness_complete']
        assert all(o.calls==1 for o in owners.values())
        rows.append('partial-'+('+'.join(names) or 'before-renderer'))
    env=env_class.__new__(env_class)
    env._offline_renderer=Owner();env.recorder_manager=Owner()
    result=close_environment(env,True)
    assert result['status']=='FULL_ENV_CLOSE_RETURNED'
    assert env._offline_renderer.calls==env.recorder_manager.calls==1
    rows.append('fully-initialized-real-close')
    for initialized in (False,True):
        env=env_class.__new__(env_class)
        env._offline_renderer=None;env.recorder_manager=Owner(fail=True)
        try:close_environment(env,initialized)
        except ValueError as exc:assert str(exc)=='cleanup-fixture-failure'
        else:raise AssertionError('Real cleanup failure was hidden')
        assert env.recorder_manager.calls==1
        rows.append('cleanup-error-propagates-'+str(initialized))
    return dict(passed=True,cases=rows,real_pinned_environment_close=True,
                environment_constructions=0,native_integrations=0,fixture_owners_only=True)
