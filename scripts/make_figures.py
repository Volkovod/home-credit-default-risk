"""Rebuild publication figures using only committed aggregate experiment records."""
from pathlib import Path
import argparse
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT/'reports/metrics'
FIGURES = ROOT/'reports/figures'
COLORS = {'teal': '#087f8c', 'orange': '#e09132', 'ink': '#16324f', 'muted': '#758397', 'red': '#c45c5c'}


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 14,
        'figure.facecolor': '#fbfcfe', 'axes.facecolor': '#fbfcfe', 'text.color': COLORS['ink'],
        'axes.labelcolor': COLORS['ink'], 'xtick.color': COLORS['muted'], 'ytick.color': COLORS['ink'],
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#d4dce5',
        'grid.color': '#e3e8ef', 'grid.alpha': .85, 'svg.fonttype': 'none', 'savefig.dpi': 180})

    def save(fig, name):
        for suffix in ['png', 'svg']:
            fig.savefig(FIGURES/f'{name}.{suffix}', bbox_inches='tight')
        plt.close(fig)

    summary = json.loads((METRICS/'run_summary.json').read_text())
    current = pd.read_csv(METRICS/'benchmark.csv')
    initial = pd.read_csv(METRICS/'initial_benchmark.csv')
    rows = pd.concat([initial, current]).drop_duplicates('name', keep='last')
    order = ['lgb_cpu_baseline', 'lgb_gpu_63', 'xgb_gpu_depth7', 'cat_gpu_depth6',
             'xgb_aug_depth7', 'xgb_peer_depth7', 'xgb_peer_depth4', 'lgb_peer_goss', 'lgb_peer_gbdt']
    labels = ['LightGBM · CPU reference', 'LightGBM · GPU / 63 bins', 'XGBoost · original / depth 7',
              'CatBoost · original / depth 6', 'XGBoost · +302 features', 'XGBoost · +40 peer features',
              'XGBoost · depth 4 / revised parameters', 'LightGBM · peer / GOSS', 'LightGBM · peer / GBDT']
    frame = rows.set_index('name').loc[order]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.7), gridspec_kw={'width_ratios':[1.45, 1]}, layout='constrained')
    color = [COLORS['teal'] if n == 'xgb_peer_depth4' else COLORS['muted'] for n in order]
    ax.scatter(frame.auc, np.arange(len(frame)), c=color, s=70, zorder=3)
    ax.set_yticks(np.arange(len(frame)), labels)
    ax.invert_yaxis()
    ax.set_xlim(.7868, .7934)
    ax.set_xlabel('ROC AUC · identical 49,202 validation clients')
    ax.set_title('Validation ROC AUC', loc='left', pad=18)
    ax.grid(axis='x')
    for i, value in enumerate(frame.auc):
        ax.annotate(f'{value:.5f}', (value, i), xytext=(8, 0), textcoords='offset points', va='center', fontsize=9)
    bx.barh(np.arange(len(frame)), frame.fit_seconds, color=color, alpha=.85)
    bx.set_yticks(np.arange(len(frame)), [''] * len(frame))
    bx.invert_yaxis()
    bx.set_xlim(0, 450)
    bx.set_title('Training time', loc='left', pad=18)
    bx.set_xlabel('Reported fit seconds · includes model export')
    for i, value in enumerate(frame.fit_seconds):
        bx.text(value + 5, i, f'{value:.0f}s', va='center', fontsize=9)
    fig.suptitle('Home Credit · model comparison', fontsize=18, fontweight='bold', x=.03, ha='left')
    save(fig, 'benchmark')

    paired = json.loads((METRICS/'paired_feature_comparison.json').read_text())
    fig, ax = plt.subplots(figsize=(10.5, 4.2), layout='constrained')
    lab = ['+302 features · same XGB parameters', '+302 features + peers · same XGB parameters',
           'Peers + XGB depth 4 / revised parameters¹', 'New blend vs previous blend²']
    for i, r in enumerate(paired):
        lo, hi = r['ci95']; v = r['delta_auc']
        ax.errorbar(v, i, xerr=[[v-lo], [hi-v]], fmt='o', capsize=5, color=COLORS['teal'], lw=2)
    ax.axvline(0, color=COLORS['orange'], linestyle='--')
    ax.set_yticks(range(len(lab)), lab)
    ax.invert_yaxis()
    ax.grid(axis='x')
    ax.set_xlabel('Paired Δ ROC AUC · 300 stratified bootstrap samples · 95% interval')
    ax.set_title('Feature and parameter comparisons', loc='left', pad=18, fontweight='bold')
    fig.text(.02, -.07, '¹ Several hyperparameters change together.  ² Models and weights were selected on this benchmark.\nIntervals condition on trained predictions; selection uncertainty is not included.', fontsize=9, color=COLORS['muted'])
    save(fig, 'ablation')

    roc = pd.read_csv(METRICS/'holdout_roc.csv')
    pr = pd.read_csv(METRICS/'holdout_pr.csv')
    cal = pd.read_csv(METRICS/'holdout_calibration.csv')
    data = json.loads((METRICS/'data_summary.json').read_text())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), layout='constrained')
    ax = axes[0,0]
    ax.plot(roc.fpr, roc.tpr, color=COLORS['teal'], lw=2.5, label=f"AUC = {summary['holdout']['auc']:.5f}")
    ax.plot([0,1],[0,1], '--', color=COLORS['muted'], lw=1)
    ax.set(xlabel='False positive rate', ylabel='True positive rate', title='Discrimination · ROC')
    ax.legend(frameon=False)
    ax = axes[0,1]
    ax.plot(pr.recall, pr.precision, color=COLORS['teal'], lw=2.5, label=f"AP = {summary['holdout']['ap']:.5f}")
    ax.axhline(data['holdout_prevalence'], color=COLORS['orange'], ls='--', label='Positive prevalence')
    ax.set(xlabel='Recall', ylabel='Precision', title='Imbalanced target · precision / recall')
    ax.legend(frameon=False)
    ax = axes[1,0]
    ax.plot(cal.mean_prediction, cal.observed_rate, 'o-', color=COLORS['teal'], lw=2)
    upper = max(cal.mean_prediction.max(), cal.observed_rate.max()) * 1.1
    ax.plot([0,upper],[0,upper],'--',color=COLORS['muted'], lw=1)
    ax.set(xlabel='Mean predicted probability', ylabel='Observed positive rate', title='Calibration · ten equal-size groups')
    ax = axes[1,1]
    ax.bar(cal.decile, cal.lift, color=[COLORS['teal']]*9+[COLORS['orange']])
    ax.axhline(1, color=COLORS['muted'], ls='--')
    ax.set(xlabel='Predicted risk decile · low → high', ylabel='Positive rate / overall prevalence', title='Risk ranking · lift by decile', xticks=range(1,11))
    for ax in axes.flat:
        ax.grid(alpha=.45)
        ax.set_axisbelow(True)
    fig.suptitle('Holdout results · 61,502 clients', fontsize=16, fontweight='bold')
    save(fig, 'holdout')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), layout='constrained')
    for ax, model, title in zip(axes, ['xgb_peer_depth4', 'cat_gpu_depth6'], ['XGBoost · normalized split gain', 'CatBoost · PredictionValuesChange']):
        imp = pd.read_csv(METRICS/f'final_{model}_42_importance.csv').head(12).iloc[::-1]
        labels = [re.sub(r'^f\d+_', '', x).replace('__',' · ') for x in imp.feature]
        labels = [x if len(x)<49 else x[:46]+'…' for x in labels]
        ax.barh(labels, imp.importance, color=COLORS['teal'], alpha=.85)
        ax.tick_params(axis='y', labelsize=8)
        ax.set_title(title, loc='left', pad=14)
        ax.set_xlabel('Importance · model-specific scale')
        ax.grid(axis='x'); ax.set_axisbelow(True)
    fig.suptitle('Feature importance · final models', fontsize=15, fontweight='bold')
    save(fig, 'feature_importance')
    print('Created four figures in PNG and SVG formats.')


if __name__ == '__main__':
    main()
