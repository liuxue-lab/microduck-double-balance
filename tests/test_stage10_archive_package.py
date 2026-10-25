"""Keep archived recovery checks importable without laptop evidence or a GPU."""
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RECOVERY = ROOT / 'scripts/stage10_diagnostics/initialization_recovery'


def test_recovery_archive_import_and_package_integrity_without_laptop():
    # A fresh process catches missing import-time JSON and module-name collisions.
    # verify_package checks the complete source/metadata inventory. It neither
    # reads a laptop trace nor constructs a simulator or imports torch.
    result = subprocess.run(
        [sys.executable, '-c',
         'import sys; import stage10_recovery_common as common; '
         'import test_recovery; common.verify_package(); '
         'assert "torch" not in sys.modules; '
         'assert "mujoco" not in sys.modules; '
         'assert "mjlab" not in sys.modules; '
         'print("ARCHIVE_IMPORT_AND_INVENTORY_PASS")'],
        cwd=RECOVERY, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'ARCHIVE_IMPORT_AND_INVENTORY_PASS' in result.stdout
