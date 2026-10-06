"""Recreate report figures from committed study records; never reruns training.

Run with the project's Python environment. Outputs are saved beside this script.
"""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
state = json.loads((ROOT / 'study.json').read_text())
assert state['status'] == 'complete'
plan = state['plan']
seeds = plan['seeds']
conditions = [c['id'] for c in plan['conditions']]
labels = ['Gaussian σ=0.05', 'Gaussian σ=0.15', 'Gaussian σ=0.45',
          'Uniform reset', 'Random search']
short = ['Gaussian\nσ=0.05', 'Gaussian\nσ=0.15', 'Gaussian\nσ=0.45',
         'Uniform\nreset', 'Random\nsearch']
colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#666666']
styles = ['-', '-', '-', '-', '--']
histories, scores = [], []
for name in conditions:
    cohort, ends = [], []
    for seed in seeds:
        trial = state['trials'][f'{name}_seed_{seed}']
        assert trial['complete']
        rows = {}
        for attempt in trial['attempts']:
            run = ROOT / attempt
            committed = json.loads((run / 'checkpoint.json').read_text())['evaluations']
            with (run / 'history.csv').open(newline='') as stream:
                for raw in csv.DictReader(stream):
                    row = {k: float(v) for k, v in raw.items()}
                    if row['evaluations'] <= committed:
                        rows[int(row['generation'])] = row
        assert sorted(rows) == list(range(plan['generations'] + 1))
        history = [rows[g] for g in sorted(rows)]
        assert all(r['evaluations'] == (g + 1) * plan['population']
                   for g, r in enumerate(history))
        assert np.isclose(history[-1]['best'], trial['result']['fitness'])
        cohort.append(history)
        ends.append(trial['result']['fitness'])
    histories.append(cohort)
    scores.append(ends)
scores = np.asarray(scores)
x = np.array([r['evaluations'] for r in histories[0][0]])
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.titlesize': 11, 'axes.labelsize': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42,
                     'savefig.facecolor': 'white'})

def clean(ax):
    ax.grid(axis='y', color='#E2E5E8', linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, color='#777777')

def save(fig, name):
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)

fig, (ax, final) = plt.subplots(1, 2, figsize=(12, 4.6),
                               gridspec_kw={'width_ratios': [1.15, 1]})
fig.subplots_adjust(left=.07, right=.985, bottom=.19, top=.80, wspace=.23)
handles = []
for i, cohort in enumerate(histories):
    values = np.array([[r['best'] for r in history] for history in cohort])
    mean, sd = values.mean(0), values.std(0, ddof=1)
    line, = ax.plot(x, mean, color=colors[i], ls=styles[i], lw=1.8, label=labels[i])
    ax.fill_between(x, mean-sd, mean+sd, color=colors[i], alpha=.10, linewidth=0)
    handles.append(line)
ax.set(title='(a) Progress during search', xlabel='Search evaluations per run',
       ylabel='Best fitness (lower is better)', xlim=(0,6432), ylim=(4.85,5.96))
ax.set_xticks([0,1600,3200,4800,6432])
positions = np.arange(5)
for seed_index in range(len(seeds)):
    final.plot(positions, scores[:,seed_index], color='#B9BDC1', lw=.8, alpha=.75, zorder=1)
for i, values in enumerate(scores):
    final.scatter(np.full(len(values),i), values, color=colors[i], s=29,
                  edgecolors='white', linewidths=.5, zorder=3)
    final.errorbar(i+.16, values.mean(), yerr=values.std(ddof=1), fmt='s',
                   color='#20262D', markersize=4, capsize=4, lw=1.2, zorder=4)
final.set(title='(b) Final performance across five seeds',
          ylabel='Final best fitness (lower is better)', ylim=(4.85,5.96), xlim=(-.35,4.45))
final.set_xticks(positions,short)
final.legend(handles=[Line2D([],[],marker='o',ls='',color='#777777',label='Individual run'),
                      Line2D([],[],marker='s',ls='-',color='#20262D',label='Mean ± SD')],
             loc='upper right', frameon=False, fontsize=8)
for a in (ax,final): clean(a)
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5,.98),
           ncol=5, frameon=False, columnspacing=1.4, handlelength=2)
fig.text(.5,.025,'Five matched seeds • 6,432 evaluations/run • Bands and error bars: sample SD • Grey lines connect the same seed',
         ha='center',fontsize=9,color='#48515A')
save(fig,'01_performance')

fig, ax = plt.subplots(figsize=(8,4.6))
fig.subplots_adjust(left=.105,right=.97,bottom=.17,top=.77)
for i, cohort in enumerate(histories):
    values=np.array([[r['mean_gene_std'] for r in history] for history in cohort])
    mean=values.mean(0)
    for run_values in values:
        ax.plot(x,run_values,color=colors[i],ls=styles[i],lw=.65,alpha=.20)
    ax.plot(x,mean,color=colors[i],ls=styles[i],lw=1.8,label=labels[i])
ax.set(yscale='log',xlim=(0,6432),ylim=(.008,2.5),
       xlabel='Search evaluations per run',ylabel='Mean per-gene population SD (log scale)')
ax.set_xticks([0,1600,3200,4800,6432])
ax.set_yticks([.01,.03,.1,.3,1,2],['0.01','0.03','0.1','0.3','1','2'])
clean(ax)
fig.legend(loc='upper center',bbox_to_anchor=(.5,.98),ncol=3,frameon=False,fontsize=9)
fig.text(.5,.025,'Bold: mean of five seeds • Faint: individual runs • Random search: retained best population',
         ha='center',fontsize=9,color='#48515A')
save(fig,'02_genetic_diversity')
print('Verified 25 histories, including merged resume attempts. Created PNG, SVG and PDF figures.')
print('Final means:', scores.mean(axis=1))
print('Final sample SDs:', scores.std(axis=1,ddof=1))
