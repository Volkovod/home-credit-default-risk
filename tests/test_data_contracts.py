"""Regression tests for leakage-sensitive transformations in the actual scripts."""
import ast
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from analyze_predictions import verify_submission


def load_functions(filename, names, namespace):
    """Execute only selected definitions, without starting a training job."""
    tree = ast.parse((ROOT/'src'/filename).read_text(encoding='utf-8'))
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {f.name for f in functions} == set(names)
    code = ast.Module(body=functions, type_ignores=[])
    exec(compile(code, filename, 'exec'), namespace)
    return namespace


class PaymentTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_functions('build_features.py', ['ratio', 'installment_level'], {'pd': pd, 'np': np})

    def frame(self, dates, amounts):
        return pd.DataFrame({'SK_ID_CURR': [1]*len(dates), 'SK_ID_PREV': [7]*len(dates),
          'NUM_INSTALMENT_VERSION': [1]*len(dates), 'NUM_INSTALMENT_NUMBER': [1]*len(dates),
          'DAYS_INSTALMENT': [-10]*len(dates), 'AMT_INSTALMENT': [100.]*len(dates),
          'DAYS_ENTRY_PAYMENT': dates, 'AMT_PAYMENT': amounts})

    def test_partial_payments_complete_only_when_sum_reaches_due(self):
        rows, conflicts = self.ns['installment_level'](self.frame([-12, -5], [40., 60.]))
        self.assertEqual(len(conflicts), 0)
        self.assertEqual(rows.iloc[0].PAID, 100.)
        self.assertEqual(rows.iloc[0].PAID_ON_TIME, 40.)
        self.assertEqual(rows.iloc[0].DAYS_LATE_COMPLETION, 5.)

    def test_payment_after_application_is_not_observed(self):
        rows, _ = self.ns['installment_level'](self.frame([-12, 2], [40., 60.]))
        self.assertEqual(rows.iloc[0].PAID, 40.)
        self.assertEqual(rows.iloc[0].UNDERPAID, 1.)
        self.assertTrue(pd.isna(rows.iloc[0].COMPLETED_DAY))

    def test_unknown_payment_does_not_become_nonpayment(self):
        rows, _ = self.ns['installment_level'](self.frame([-12, np.nan], [40., 60.]))
        self.assertTrue(pd.isna(rows.iloc[0].UNDERPAID))

    def test_conflicting_installment_is_quarantined(self):
        raw = self.frame([-12, -5], [40., 60.])
        raw.loc[1, 'AMT_INSTALMENT'] = 200.
        rows, conflicts = self.ns['installment_level'](raw)
        self.assertEqual(len(rows), 0)
        self.assertEqual(len(conflicts), 1)

    def test_zero_denominator_stays_missing(self):
        result = self.ns['ratio'](pd.Series([10., 10.]), pd.Series([0., 2.]))
        self.assertTrue(pd.isna(result.iloc[0]))
        self.assertEqual(result.iloc[1], 5.)


class PeerTests(unittest.TestCase):
    def setUp(self):
        self.groups = ['NAME_EDUCATION_TYPE','CODE_GENDER','OCCUPATION_TYPE','ORGANIZATION_TYPE',
                       'REGION_RATING_CLIENT','NAME_INCOME_TYPE']
        self.values = ['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','EXT_SOURCE_2','EXT_SOURCE_3']
        self.train = pd.DataFrame({c: ['A']*60 for c in self.groups})
        for c in self.values:
            self.train[c] = np.arange(1., 61.)
        self.ns = load_functions('train_research.py', ['cohort_key', 'prepare'],
                 {'pd': pd, 'np': np, 'name_map': {c:c for c in self.train},
                  'original_cols': list(self.train), 'X': self.train})
        self.spec = {'features':'peer', 'kind':'xgb'}

    def test_validation_values_cannot_change_fitted_state(self):
        valid = self.train.iloc[:2].copy()
        a, _, state_a = self.ns['prepare'](self.train, [valid], self.spec)
        valid['AMT_INCOME_TOTAL'] = 1e12
        b, _, state_b = self.ns['prepare'](self.train, [valid], self.spec)
        pd.testing.assert_frame_equal(a[0], b[0])
        self.assertEqual(state_a, state_b)

    def test_unseen_peer_uses_training_median_and_unknown_category(self):
        valid = self.train.iloc[:1].copy()
        valid['NAME_EDUCATION_TYPE'] = 'UNSEEN'
        frames, _, state = self.ns['prepare'](self.train, [valid], self.spec)
        self.assertTrue(pd.isna(frames[1].NAME_EDUCATION_TYPE.iloc[0]))
        col = 'PEER_NAME_EDUCATION_TYPE_CODE_GENDER__AMT_INCOME_TOTAL_DIFF'
        self.assertAlmostEqual(frames[1][col].iloc[0], 1. - 30.5)
        self.assertEqual(len(frames[0].columns), len(self.train.columns) + 40)


class SubmissionTests(unittest.TestCase):
    def test_reordered_ids_fail_even_if_sets_match(self):
        with self.assertRaises(ValueError):
            verify_submission(pd.DataFrame({'SK_ID_CURR':[2,1], 'TARGET':[.2,.1]}), [1,2])

    def test_missing_probability_fails(self):
        with self.assertRaises(ValueError):
            verify_submission(pd.DataFrame({'SK_ID_CURR':[1,2], 'TARGET':[.2,np.nan]}), [1,2])

    def test_valid_submission_passes(self):
        verify_submission(pd.DataFrame({'SK_ID_CURR':[1,2], 'TARGET':[0.,1.]}), [1,2])


if __name__ == '__main__':
    unittest.main()
