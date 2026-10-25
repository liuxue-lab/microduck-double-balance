"""Render the received Stage 07 evidence; no training or simulation."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
evaluations = json.loads((root / 'docs/audits/stage-07-nominal-evaluation-progress.json').read_text())['rows']
diagnostics = json.loads((root / 'docs/audits/stage-07-local-diagnostic.json').read_text())
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
for name, label, color in [('train-20260927T103517Z-3844', 'Original segment', '#1768AC'),
                            ('train-20260927T144559Z-1633', 'Resumed segment', '#D46624')]:
    rows = [r for r in evaluations if r['run'] == name]
    axes[0].plot([r['completed_updates'] for r in rows], [100*r['mean_stable_fraction'] for r in rows],
                 'o-', color=color, label=label, markersize=5)
axes[0].set(title='Cloud nominal evaluation: stability declines', xlabel='Completed PPO updates',
            ylabel='Mean stable time (%)', ylim=(0, 100))
axes[0].axvline(1000, color='#1768AC', linestyle=':', alpha=.55)
axes[0].legend(frameon=False)
axes[0].grid(alpha=.2)
names = ['lower_ball_speed', 'top_speed', 'top_center']
x = np.arange(len(names))
for offset, role, label, color in [(-.18, 'best_nominal', 'Update 1000', '#1768AC'),
                                   (.18, 'final_state', 'Update 6000', '#D46624')]:
    values = [100*diagnostics[role]['failure_fraction_per_condition'][name] for name in names]
    bars = axes[1].bar(x + offset, values, .36, label=label, color=color)
    axes[1].bar_label(bars, fmt='%.2f', fontsize=9, padding=3)
axes[1].set(title='Local diagnostic: lower-ball speed dominates', ylabel='Environment steps violating condition (%)',
            xticks=x, xticklabels=['Lower-ball speed\n> 0.15 m/s', 'Top-ball relative speed\n> 0.08 m/s',
                                   'Top-ball offset\n> 12 mm'], ylim=(0, 72))
axes[1].legend(frameon=False)
axes[1].grid(axis='y', alpha=.2)
fig.suptitle('Stage 07: 6000 updates completed; strict task success remains 0%', fontsize=13)
output = root / 'docs/audits/figures/stage-07-evaluation-and-diagnosis.png'
fig.savefig(output, dpi=160)
print(output)
