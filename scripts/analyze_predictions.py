"""Verify downloaded predictions and export aggregate evidence without client rows."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, precision_recall_curve, roc_auc_score, roc_curve


def measures(y, p):
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Predictions must be finite probabilities.')
    return {'auc': float(roc_auc_score(y, p)), 'ap': float(average_precision_score(y, p)),
            'logloss': float(log_loss(y, p))}


def verify_submission(frame, expected_ids):
    if list(frame) != ['SK_ID_CURR', 'TARGET']:
        raise ValueError('Unexpected submission columns.')
    if not frame.SK_ID_CURR.is_unique or not np.array_equal(frame.SK_ID_CURR, expected_ids):
        raise ValueError('Submission IDs or their order differ from the template.')
    p = frame.TARGET.to_numpy()
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid submission probabilities.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--sample-submission', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('reports/metrics'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def save(name, value):
        (args.output/name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

    source = args.result_dir
    summary = json.loads((source/'run_summary.json').read_text())
    frozen = json.loads((source/'frozen_selection.json').read_text())
    assert frozen['chosen']['weights'] == summary['weights']
    oof, hold, bench, split = [pd.read_csv(source/f'{name}.csv') for name in
                              ['oof_predictions', 'holdout_predictions', 'benchmark_predictions', 'split_manifest']]
    assert len(oof) == 246009 and len(hold) == 61502 and len(bench) == 49202
    assert oof.SK_ID_CURR.is_unique and hold.SK_ID_CURR.is_unique and split.SK_ID_CURR.is_unique
    assert set(oof.SK_ID_CURR).isdisjoint(hold.SK_ID_CURR)
    assert set(oof.SK_ID_CURR) == set(split.loc[split.split.eq('development'), 'SK_ID_CURR'])
    assert set(hold.SK_ID_CURR) == set(split.loc[split.split.eq('holdout'), 'SK_ID_CURR'])
    assert set(bench.SK_ID_CURR) <= set(oof.SK_ID_CURR)
    assert set(oof.fold.unique()) == set(range(5))
    assert np.array_equal(bench.TARGET, oof.set_index('SK_ID_CURR').loc[bench.SK_ID_CURR, 'TARGET'])
    checked = {}
    for name, frame in [('oof', oof), ('holdout', hold)]:
        blend = sum(w * frame[model].to_numpy() for model, w in summary['weights'].items())
        np.testing.assert_allclose(frame.prediction, blend, rtol=0, atol=1e-12)
        result = measures(frame.TARGET, frame.prediction)
        for metric, value in result.items():
            assert abs(value - summary[name][metric]) < 1e-10, (name, metric)
        checked[name] = result
    for row in summary['benchmark_ranking']:
        assert abs(roc_auc_score(bench.TARGET, bench[row['name']]) - row['auc']) < 1e-10
    for model in summary['weights']:
        folds = pd.read_csv(source/f'cv_metrics_{model}.csv')
        for row in folds.to_dict('records'):
            part = oof.loc[oof.fold.eq(row['fold'])]
            assert abs(roc_auc_score(part.TARGET, part[model]) - row['auc']) < 1e-10

    template = pd.read_csv(args.sample_submission)
    assert len(template) == 48744 and template.SK_ID_CURR.is_unique
    assert set(template.SK_ID_CURR).isdisjoint(split.SK_ID_CURR)
    submissions = {}
    for name in ['submission.csv', 'submission_cv.csv']:
        frame = pd.read_csv(source/name)
        verify_submission(frame, template.SK_ID_CURR)
        submissions[name] = {'rows': len(frame), 'min': float(frame.TARGET.min()),
                             'max': float(frame.TARGET.max()), 'sha256': hashlib.sha256((source/name).read_bytes()).hexdigest()}
    save('verification.json', {'passed': True, 'checks': ['metrics recomputed', 'five fold metrics recomputed',
         'frozen weighted predictions', 'disjoint row sets', 'benchmark contained in development',
         'submission IDs and order equal official template', 'finite bounded probabilities'],
         'metrics': checked, 'submissions': submissions})

    # Paired resampling keeps the same clients for both candidates.
    y = bench.TARGET.to_numpy()
    old = .75 * bench.xgb_gpu_depth7.to_numpy() + .25 * bench.cat_gpu_depth6.to_numpy()
    new = .75 * bench.xgb_peer_depth4.to_numpy() + .25 * bench.cat_gpu_depth6.to_numpy()
    pairs = [(n, bench[n].to_numpy(), 'original XGB depth 7', bench.xgb_gpu_depth7.to_numpy())
             for n in ['xgb_aug_depth7', 'xgb_peer_depth7', 'xgb_peer_depth4']]
    pairs.append(('new_selected_blend', new, 'previous 75/25 blend', old))
    samples = {n: [] for n, *_ in pairs}
    rng = np.random.default_rng(42)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for _ in range(300):
        ix = np.r_[rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        for name, p, _, reference in pairs:
            samples[name].append(roc_auc_score(y[ix], p[ix]) - roc_auc_score(y[ix], reference[ix]))
    paired = [{'model': n, 'reference': ref, 'delta_auc': float(roc_auc_score(y, p) - roc_auc_score(y, base)),
               'ci95': np.quantile(samples[n], [.025, .975]).tolist()} for n, p, ref, base in pairs]
    save('paired_feature_comparison.json', paired)

    data = {'train_rows': len(split), 'test_rows': len(template), 'development_rows': len(oof),
            'holdout_rows': len(hold), 'benchmark_rows': len(bench),
            'target_positive': int(oof.TARGET.sum() + hold.TARGET.sum()),
            'development_prevalence': float(oof.TARGET.mean()), 'holdout_prevalence': float(hold.TARGET.mean()),
            'oof_prediction_correlation': float(oof[list(summary['weights'])].corr().iloc[0, 1]),
            'oof_residual_correlation': float(oof[list(summary['weights'])].sub(oof.TARGET, axis=0).corr().iloc[0, 1])}
    save('data_summary.json', data)
    grid = np.linspace(0, 1, 301)
    fpr, tpr, _ = roc_curve(hold.TARGET, hold.prediction)
    precision, recall, _ = precision_recall_curve(hold.TARGET, hold.prediction)
    pd.DataFrame({'fpr': grid, 'tpr': np.interp(grid, fpr, tpr)}).to_csv(args.output/'holdout_roc.csv', index=False)
    pd.DataFrame({'recall': grid, 'precision': np.interp(grid, recall[::-1], precision[::-1])}).to_csv(args.output/'holdout_pr.csv', index=False)
    # Quantile diagnostics are descriptive only; they do not change predictions.
    ranked = hold.sort_values('prediction').copy()
    ranked['decile'] = np.minimum(np.arange(len(ranked)) * 10 // len(ranked), 9) + 1
    calibration = ranked.groupby('decile').agg(n=('TARGET','size'), mean_prediction=('prediction','mean'),
                                              observed_rate=('TARGET','mean'), positives=('TARGET','sum')).reset_index()
    calibration['lift'] = calibration.observed_rate / hold.TARGET.mean()
    calibration.to_csv(args.output/'holdout_calibration.csv', index=False)
    pd.DataFrame([{'fold': int(f), 'n': len(g), 'positives': int(g.TARGET.sum()),
                  'auc_blend': float(roc_auc_score(g.TARGET, g.prediction))} for f, g in oof.groupby('fold')]).to_csv(args.output/'fold_summary.csv', index=False)
    print(json.dumps({'verified': checked, 'paired': paired, 'data': data}, indent=2))


if __name__ == '__main__':
    main()
