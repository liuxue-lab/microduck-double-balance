"""Stage12 journal totals; G11 ledger format and durability remain unchanged."""
import time
import shutil
from runtime_accounting import Journal, save


class SessionJournal(Journal):
    def __init__(self, out, counts, caps, plan):
        self.plan = plan
        super().__init__(out, counts, caps)

    def progress(self, force=False):
        from integration.cloud_limits import check_boundary
        check_boundary(self.out, force=force)
        now = time.monotonic()
        if force or now - self.last_progress >= 5:
            if not force and not getattr(self, 'disk_limit_seen', False) and shutil.disk_usage(self.out).free < 2 * 1024**3:
                self.disk_limit_seen = True  # Let cleanup/final receipts use reserved free space.
                raise RuntimeError('Evidence disk free space below 2 GiB; stop and preserve WAL')
            save(self.out / 'progress.json', dict(
                phase=self.phase, elapsed_s=now-self.start,
                logical_world=self.plan['logical_world'],
                macro_completed=self.macros, macro_cap=self.plan['max_macros'],
                formal_steps_completed=self.formal_steps,
                formal_steps_cap=self.plan['public_steps'], counts=self.counts,
                caps=self.caps, native_WAL_pending=dict(self.pending),
                stage11='11/11 CLOSED_SCOPED', PPO=0, cloud_training=0,
                native_added_completed=self.counts.get('integrations_completed', 0),
                native_added_attempted=self.counts.get('integrations_attempted', 0),
                canonical_base={'completed':704664,'attempted':704665},
                canonical_must_be_aggregated_once_across_owners=True,
                wall_deadline=__import__("os").environ.get("MD12_NATIVE_DEADLINE")))
            self.last_progress = now
