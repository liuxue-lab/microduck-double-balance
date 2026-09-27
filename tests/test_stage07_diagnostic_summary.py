"""Check that diagnostic streaks do not confuse average/ever/final stability."""
from pathlib import Path
import runpy
import unittest

streaks = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/diagnose_stage07_local.py'))['streaks']


class DiagnosticStreakTests(unittest.TestCase):
    def test_high_stable_fraction_can_have_no_five_second_streak(self):
        flags = ([True] * 45 + [False] * 5) * 10
        self.assertEqual(sum(flags) / len(flags), .9)
        self.assertEqual(streaks(flags), (45, 0))

    def test_earlier_success_does_not_imply_final_success(self):
        self.assertEqual(streaks([True] * 300 + [False] + [True] * 199), (300, 199))

    def test_terminal_streak_kept_in_control_steps(self):
        self.assertEqual(streaks([False] * 200 + [True] * 300), (300, 300))


if __name__ == '__main__':
    unittest.main()
