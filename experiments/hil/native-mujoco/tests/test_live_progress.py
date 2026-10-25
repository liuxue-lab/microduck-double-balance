"""Live counters never invoke durable file writes in the control worker."""
import io
import unittest
from unittest.mock import patch
import live_progress

class LiveProgress(unittest.TestCase):
    def test_stream_roundtrip_without_file_or_directory_fsync(self):
        row=dict(status='RUNNING',case_number=2,cases_total=2,percent=75.,detail='candidate_first_episode',
            total_control_steps=6000,completed_control_steps=4500,elapsed_seconds=90.,
            control_steps=1500,checked_physics_steps=6000,completed_pulses=14)
        stream=io.StringIO()
        with patch('os.fsync',side_effect=AssertionError('Control progress must not fsync')), \
                patch('builtins.open',side_effect=AssertionError('No control progress file transaction')):
            live_progress.emit(row,stream)
        self.assertEqual(live_progress.parse(stream.getvalue()),row)
        self.assertIn('4500/6000',live_progress.display(row))
        self.assertIsNone(live_progress.parse('ordinary runtime log'))
        with self.assertRaises(RuntimeError):live_progress.parse(live_progress.PREFIX+'{}')

    def test_partial_log_writes_are_not_parsed_until_newline(self):
        source='ordinary log\n'+live_progress.PREFIX+'{"total_control_steps":6000,"completed_control_steps":25}\n'
        pending='';rows=[]
        for character in source:
            ready,pending=live_progress.complete_lines(pending,character)
            rows.extend(ready)
        self.assertEqual(pending,'')
        self.assertEqual(len(rows),2)
        self.assertEqual(live_progress.parse(rows[1])['completed_control_steps'],25)
