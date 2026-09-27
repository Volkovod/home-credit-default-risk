"""Target-free Home Credit feature extension. Original implementation.

Ideas: js-aguiar/home-credit-default-competition (loan conditions, recent windows),
NoxMoon/home-credit-default-risk (account-level statistics, feature diversity).
No published predictions, targets, or fitted models are used.
"""
from pathlib import Path
import gc, hashlib, json, os, shutil, time
import numpy as np
import pandas as pd

START = time.time()
OUT = Path(os.environ.get('HC_OUTPUT', '/kaggle/working/home_credit_research_features'))
OUT.mkdir(parents=True, exist_ok=True)
INPUT = Path(os.environ.get('HC_INPUT_ROOT', '/kaggle/input'))
CACHE = Path(os.environ['HC_CACHE']) if 'HC_CACHE' in os.environ else next(
    p.parent for p in INPUT.rglob('manifest.json')
    if json.loads(p.read_text()).get('feature_version') == 'original-v2.2-cache-v1')
RAW = Path(os.environ['HC_RAW']) if 'HC_RAW' in os.environ else next(INPUT.rglob('application_train.csv')).parent
manifest = json.loads((CACHE/'manifest.json').read_text())
labels = pd.read_parquet(CACHE/'train_labels.parquet')
test_ids = pd.read_parquet(CACHE/'test_ids.parquet')
ids = pd.Index(pd.concat([labels.SK_ID_CURR, test_ids.SK_ID_CURR]), name='SK_ID_CURR')
assert ids.is_unique
blocks = {}

def ratio(a, b):
    return a / b.where(b.abs() > 1e-9)

def aggregate(frame, prefix, cols, funcs=('mean', 'max', 'std')):
    if frame.empty:
        return pd.DataFrame(index=pd.Index([], name='SK_ID_CURR'))
    g = frame.groupby('SK_ID_CURR', observed=True, sort=False)
    out = g[list(cols)].agg(list(funcs))
    out.columns = [f'R2_{prefix}__{c}__{f}' for c, f in out.columns]
    out[f'R2_{prefix}__COUNT'] = g.size()
    return out.astype('float32')

def keep(block, name):
    assert block.index.is_unique
    block = block.reindex(ids).replace([np.inf, -np.inf], np.nan).astype('float32')
    blocks[name] = block
    print('BLOCK', name, block.shape, 'seconds', round(time.time()-START, 1), flush=True)

app = pd.concat([pd.read_csv(RAW/'application_train.csv').drop(columns='TARGET'),
                 pd.read_csv(RAW/'application_test.csv')]).set_index('SK_ID_CURR').loc[ids]
a = pd.DataFrame(index=ids)
ext = app[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']]
a['R2_APP__EXT_PRODUCT_ALL'] = ext.prod(axis=1, min_count=3)
a['R2_APP__EXT_WEIGHTED'] = 2*ext.EXT_SOURCE_1 + 3*ext.EXT_SOURCE_2 + 4*ext.EXT_SOURCE_3
a['R2_APP__CREDIT_DOWNPAYMENT_PROXY'] = app.AMT_GOODS_PRICE - app.AMT_CREDIT
a['R2_APP__INCOME_AFTER_ANNUITY'] = app.AMT_INCOME_TOTAL/12 - app.AMT_ANNUITY
a['R2_APP__CHILDREN_SHARE'] = ratio(app.CNT_CHILDREN, app.CNT_FAM_MEMBERS)
a['R2_APP__CREDIT_PER_FAMILY'] = ratio(app.AMT_CREDIT, app.CNT_FAM_MEMBERS)
a['R2_APP__ANNUITY_PER_FAMILY'] = ratio(app.AMT_ANNUITY, app.CNT_FAM_MEMBERS)
docs = app.filter(regex='^FLAG_DOCUMENT_')
a['R2_APP__DOCUMENT_KURTOSIS'] = docs.kurt(axis=1)
housing = [c for c in app if c.endswith('_AVG')]
a['R2_APP__HOUSING_MISSING'] = app[housing].isna().sum(axis=1)
a['R2_APP__CONTACT_COUNT'] = app[['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE',
                                'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL']].sum(axis=1)
for ext_name in ext:
    for denom in ['AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_BIRTH']:
        a[f'R2_APP__{ext_name}_BY_{denom}'] = ratio(app[ext_name], app[denom])
keep(a, 'application')
del a

prev = pd.read_csv(RAW/'previous_application.csv')
assert prev.SK_ID_PREV.is_unique
for c in [c for c in prev if c.startswith('DAYS_')]:
    prev[c] = prev[c].replace(365243, np.nan)
prev = prev[prev.DAYS_DECISION.le(0)].copy()
prev['TERM_RATIO'] = ratio(prev.AMT_CREDIT, prev.AMT_ANNUITY)
prev['TOTAL_SCHEDULED'] = prev.AMT_ANNUITY*prev.CNT_PAYMENT
prev['TOTAL_COST_RATIO'] = ratio(prev.TOTAL_SCHEDULED, prev.AMT_CREDIT)-1
prev['SIMPLE_COST_PER_MONTH'] = ratio(prev.TOTAL_COST_RATIO, prev.CNT_PAYMENT)
prev['CREDIT_MINUS_APPLICATION'] = prev.AMT_CREDIT-prev.AMT_APPLICATION
prev['CREDIT_GOODS_RATIO'] = ratio(prev.AMT_CREDIT, prev.AMT_GOODS_PRICE)
prev['END_SCHEDULE_CHANGE'] = prev.DAYS_LAST_DUE-prev.DAYS_LAST_DUE_1ST_VERSION
prev['FIRST_TO_LAST_DUE'] = prev.DAYS_LAST_DUE_1ST_VERSION-prev.DAYS_FIRST_DUE
# Cash payments observed on/before the application date; partial payments are summed.
parts = []
for chunk in pd.read_csv(RAW/'installments_payments.csv',
                         usecols=['SK_ID_PREV','AMT_PAYMENT','DAYS_ENTRY_PAYMENT'], chunksize=700000):
    chunk = chunk.loc[chunk.DAYS_ENTRY_PAYMENT.le(0)]
    parts.append(chunk.groupby('SK_ID_PREV').AMT_PAYMENT.sum(min_count=1))
paid = pd.concat(parts).groupby(level=0).sum(min_count=1)
prev['OBSERVED_PAYMENTS'] = prev.SK_ID_PREV.map(paid)
# This is an approximate contractual cash-flow gap, NOT an observed principal balance.
prev['SCHEDULED_REPAYMENT_GAP'] = (prev.TOTAL_SCHEDULED-prev.OBSERVED_PAYMENTS).clip(lower=0)
prev['OBSERVED_REPAYMENT_RATIO'] = ratio(prev.OBSERVED_PAYMENTS, prev.TOTAL_SCHEDULED)
del paid, parts, chunk
cols = ['TERM_RATIO','TOTAL_COST_RATIO','SIMPLE_COST_PER_MONTH','CREDIT_MINUS_APPLICATION',
        'CREDIT_GOODS_RATIO','END_SCHEDULE_CHANGE','FIRST_TO_LAST_DUE',
        'SCHEDULED_REPAYMENT_GAP','OBSERVED_REPAYMENT_RATIO']
groups = [('PREV_ALL', prev),
          ('PREV_APPROVED', prev[prev.NAME_CONTRACT_STATUS.eq('Approved')]),
          ('PREV_CASH', prev[prev.NAME_CONTRACT_TYPE.eq('Cash loans')]),
          ('PREV_CONSUMER', prev[prev.NAME_CONTRACT_TYPE.eq('Consumer loans')]),
          ('PREV_90D', prev[prev.DAYS_DECISION.ge(-90)]),
          ('PREV_730D', prev[prev.DAYS_DECISION.ge(-730)])]
for name, frame in groups:
    keep(aggregate(frame, name, cols), name)
del groups, frame
last = prev.sort_values(['SK_ID_CURR','DAYS_DECISION','SK_ID_PREV']).drop_duplicates('SK_ID_CURR', keep='last')
keep(last.set_index('SK_ID_CURR')[cols+['CNT_PAYMENT']].add_prefix('R2_PREV_LAST__'), 'last_loan')
active = prev.loc[prev.NAME_CONTRACT_STATUS.eq('Approved') & prev.DAYS_FIRST_DUE.le(0)
                  & prev.DAYS_LAST_DUE_1ST_VERSION.gt(0)]
active_agg = aggregate(active, 'SCHEDULE_ACTIVE',
                       ['AMT_ANNUITY','AMT_CREDIT','SCHEDULED_REPAYMENT_GAP'], ('sum','max'))
for c in list(active_agg):
    if c.endswith('__sum'):
        active_agg[c+'_INCOME_RATIO'] = ratio(active_agg[c], app.AMT_INCOME_TOTAL.reindex(active_agg.index))
keep(active_agg, 'scheduled_active')
del prev, last, active, active_agg
gc.collect()

bureau = pd.read_csv(RAW/'bureau.csv')
assert bureau.SK_ID_BUREAU.is_unique
bureau = bureau.loc[bureau.DAYS_CREDIT.le(0)].copy()
bureau['DEBT_TO_CREDIT'] = ratio(bureau.AMT_CREDIT_SUM_DEBT, bureau.AMT_CREDIT_SUM)
bureau['DEBT_MINUS_CREDIT'] = bureau.AMT_CREDIT_SUM_DEBT-bureau.AMT_CREDIT_SUM
bureau['CLOSE_SCHEDULE_GAP'] = bureau.DAYS_ENDDATE_FACT-bureau.DAYS_CREDIT_ENDDATE
bureau['CREDIT_TERM'] = bureau.DAYS_CREDIT_ENDDATE-bureau.DAYS_CREDIT
cols = ['AMT_CREDIT_SUM','AMT_CREDIT_SUM_DEBT','DEBT_TO_CREDIT','DEBT_MINUS_CREDIT',
        'CLOSE_SCHEDULE_GAP','CREDIT_TERM','AMT_CREDIT_MAX_OVERDUE']
for name, frame in [('BUREAU_90D', bureau[bureau.DAYS_CREDIT.ge(-90)]),
                    ('BUREAU_730D', bureau[bureau.DAYS_CREDIT.ge(-730)]),
                    ('BUREAU_CONSUMER', bureau[bureau.CREDIT_TYPE.eq('Consumer credit')]),
                    ('BUREAU_CARD', bureau[bureau.CREDIT_TYPE.eq('Credit card')])]:
    keep(aggregate(frame, name, cols), name)
last = bureau.sort_values(['SK_ID_CURR','DAYS_CREDIT','SK_ID_BUREAU']).drop_duplicates('SK_ID_CURR', keep='last')
keep(last.set_index('SK_ID_CURR')[cols].add_prefix('R2_BUREAU_LAST__'), 'last_bureau')
del bureau, last, frame, app
gc.collect()

extra = pd.concat(list(blocks.values()), axis=1)
assert extra.columns.is_unique
extra.to_parquet(OUT/'extra_features.parquet', compression='zstd')
for file in ['train_labels.parquet','test_ids.parquet','original_splits.npz','sample_submission.csv']:
    shutil.copyfile(CACHE/file, OUT/file)
counts = {}
for split, row_ids in [('train', labels.SK_ID_CURR), ('test', test_ids.SK_ID_CURR)]:
    old = pd.read_parquet(CACHE/f'{split}_features.parquet')
    ext_frame = extra.loc[row_ids].reset_index(drop=True)
    assert not set(old)&set(ext_frame)
    joined = pd.concat([old, ext_frame], axis=1)
    assert len(joined)==len(row_ids) and joined.columns.is_unique
    joined.to_parquet(OUT/f'{split}_features.parquet', index=False, compression='zstd')
    counts[split] = list(joined.shape)
    del old, ext_frame, joined
    gc.collect()
manifest.update(feature_version='research-v2-20260926',parent_matrix_hash=manifest['matrix_hash'],
    original_feature_count=manifest['features'],features=counts['train'][1],
    research_blocks={k:list(v.columns) for k,v in blocks.items()},
    research_seconds=time.time()-START, research_note='Target-free per-client features; peer statistics are fitted within each model training fold.')
manifest['matrix_hash'] = hashlib.sha256((OUT/'train_features.parquet').read_bytes()).hexdigest()
(OUT/'manifest.json').write_text(json.dumps(manifest, indent=2))
print('AUGMENT_COMPLETE', json.dumps(dict(shapes=counts,added_features=extra.shape[1],seconds=time.time()-START)),flush=True)
