"""Regression tests for the shared checkpoint-rendering effects interface."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from mjlab_microduck.video_effects import CrashEffects

_ROOT = Path(__file__).resolve().parents[1]


def test_render_checkpoint_uses_supported_crash_effects_keywords() -> None:
    """Every keyword at the render call site must exist on the constructor.

    Stage 02 restored ``video_effects.py`` from a frozen upstream source, but
    the render script still passed a newer ``ground_only`` keyword.  The normal
    render smoke used effects-off mode, so only the effects path exposed the
    mismatch.  Keep the two files locked together from now on.
    """

    tree = ast.parse((_ROOT / "scripts" / "render_checkpoint.py").read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "CrashEffects"
    ]
    assert len(calls) == 1

    supported = set(inspect.signature(CrashEffects).parameters)
    used = {keyword.arg for keyword in calls[0].keywords if keyword.arg is not None}
    assert used <= supported, f"unsupported CrashEffects keywords: {used - supported}"
