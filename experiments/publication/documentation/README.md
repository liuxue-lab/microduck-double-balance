# Documentation publication tools

[Experiments](../../README.md) · **English** · [简体中文](README.zh-CN.md)

Offline Git-bundle preparation and import for the project documentation.

| File | Purpose |
| --- | --- |
| [install.py](install.py) | Import a prepared documentation update locally |
| [build_release.py](build_release.py) | Build an importer and incremental Git bundle from the selected files |
| [test_install.py](test_install.py) | Exercise import and conflict handling with temporary repositories |

Run the tool's local checks from the repository root:

```bash
python3 -m unittest discover -s experiments/publication/documentation -p 'test_install.py' -v
```

The [publication record](../../../docs/double-balance/publication.md) documents the original use of this tool.
Current source publication follows the [local SSH workflow](../../../docs/handoffs/microduck-local-ssh-push-protocol.md).
