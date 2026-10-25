"""Keep the original 2ms command-delay clock while integrating physics at 5ms.

The original DelayBuffer still samples lags 3..6 on its original 2ms clock.
The native solver samples that delayed command at 0,5,10,15ms each control.
Between samples commands use zero-order hold, never a new action filter.
"""
from dataclasses import replace
import torch

from native_backend import need


class OriginalDelayClock:
    def __init__(self, raw):
        self.raw, self.saved, self.rows = raw, [], []
        self.old_after_step = raw.sim.after_step
        need(raw.sim.step_calls == 0, 'Install delay clock only after the initial reset')
        try:
            for group in raw.scene['robot'].actuators:
                cfg = group.cfg
                need(group.command_field == 'position' and group._delay_buffer is not None,
                     'Expected original position-command delay buffer')
                need((cfg.delay_min_lag, cfg.delay_max_lag, cfg.delay_update_period,
                      cfg.delay_hold_prob) == (3, 6, 0, 0.), 'Unreviewed actuator delay recipe')
                row = dict(group=group, next_us=0, last_input=None, last_output=None, calls=0, samples=0)
                self.rows.append(row)
                self.saved.append((group, 'apply_delay' in vars(group), vars(group).get('apply_delay')))
                def apply(cmd, row=row):
                    now = round(raw.sim.mj_data.time * 1_000_000)
                    # Catch up virtual ticks before this new sample using the
                    # previous target. At 20ms, the new policy target must not
                    # retroactively enter the 16ms or 18ms delay history.
                    self.advance(row, now, inclusive=False)
                    row['last_input'] = cmd.position_target.detach().clone()
                    self.advance(row, now, inclusive=True)
                    row['calls'] += 1
                    need(row['last_output'] is not None, 'Missing delayed actuator target')
                    return replace(cmd, position_target=row['last_output'])
                group.apply_delay = apply
            raw.sim.after_step = self.after_step
        except BaseException:
            self.close()
            raise

    def advance(self, row, now, inclusive):
        limit = now if inclusive else now - 1
        while row['next_us'] <= limit:
            need(row['last_input'] is not None, 'Virtual delay clock skipped its initial command')
            buffer = row['group']._delay_buffer
            buffer.append(row['last_input'])
            row['last_output'] = buffer.compute()
            row['next_us'] += 2000
            row['samples'] += 1

    def after_step(self):
        self.old_after_step()
        if self.raw.sim.step_calls % 4 == 0:
            now = round(self.raw.sim.mj_data.time * 1_000_000)
            for row in self.rows:
                self.advance(row, now, inclusive=False)

    def snapshot(self):
        return dict(command_delay_clock_ms=2., lag_choices_ms=[6., 8., 10., 12.],
            physics_sampling_ms=5., sampling='zero-order hold of original 2ms delay output',
            native_delay_calls=[r['calls'] for r in self.rows],
            original_delay_samples=[r['samples'] for r in self.rows],
            observation_delays='unchanged at the original 20ms control cadence',
            exact_GPU_random_sequence_claimed=False)

    def close(self):
        self.raw.sim.after_step = self.old_after_step
        for group, local, previous in reversed(self.saved):
            if local:
                group.apply_delay = previous
            else:
                del group.apply_delay
        self.saved.clear()
