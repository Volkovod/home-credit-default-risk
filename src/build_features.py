from pathlib import Path
from contextlib import contextmanager
import gc, json, time, zipfile, re, importlib.metadata, hashlib, os, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
SEED=42
MODE='full'
DATA_PATH=os.environ.get('HC_RAW')
QUICK_TRAIN=30000
QUICK_TEST=5000
CHUNK_SIZE=500_000
EDA_SAMPLE=1000
N_SPLITS=3
SAVE_FEATURES=False
SAVE_FEATURE_CACHE=False
REUSE_FEATURES=False
FEATURE_VERSION='original-v2.2-cache-v1'
OUTPUT_DIR=Path(os.environ.get('HC_OUTPUT', '/kaggle/working/home_credit_features'))
REPORT_DIR=OUTPUT_DIR/'reports'
REPORT_DIR.mkdir(parents=True,exist_ok=True)
versions={p:importlib.metadata.version(p) for p in ['pandas','numpy','scikit-learn']}
START=time.time()
def display(*args,**kwargs): pass
print('FEATURE_BUILD_START',versions,flush=True)


TABLES = ['application_train','application_test','bureau','bureau_balance',
          'previous_application','POS_CASH_balance','credit_card_balance',
          'installments_payments','sample_submission','HomeCredit_columns_description']

def resolve_source(path):
    if path is not None:
        p = Path(path)
        if not p.exists(): raise FileNotFoundError(p)
        return p
    roots = [Path('/kaggle/input'), Path('./data'), Path('.')]
    for root in roots:
        if not root.exists(): continue
        found = list(root.rglob('application_train.csv'))
        if found: return found[0].parent
    for root in [Path('./data'), Path('.')]:
        if root.exists():
            found = list(root.glob('home-credit-default-risk*.zip'))
            if found: return found[0]
    raise FileNotFoundError('Нет исходных CSV. Добавьте датасет Kaggle или задайте DATA_PATH.')

SOURCE = resolve_source(DATA_PATH)
if SOURCE.is_file():
    with zipfile.ZipFile(SOURCE) as z:
        MEMBERS = {Path(x).name: x for x in z.namelist() if x.endswith('.csv')}
else:
    MEMBERS = {p.name: p for p in SOURCE.rglob('*.csv')}
missing_files = [t for t in TABLES if t + '.csv' not in MEMBERS]
assert not missing_files, f'Не хватает файлов: {missing_files}'

@contextmanager
def csv_stream(name):
    member = MEMBERS[name + '.csv']
    if SOURCE.is_file():
        with zipfile.ZipFile(SOURCE) as z:
            with z.open(member) as f: yield f
    else:
        with open(member, 'rb') as f: yield f

def read_table(name, ids=None, key='SK_ID_CURR'):
    # float64 сохраняем при чтении: важно для денежных сумм и частичных платежей.
    # Чанки фильтруются по клиентам, а не обрезаются по первым N строкам.
    pieces = []
    with csv_stream(name) as f:
        for part in pd.read_csv(f, chunksize=CHUNK_SIZE, low_memory=False):
            if ids is not None: part = part.loc[part[key].isin(ids)].copy()
            pieces.append(part)
    frame = pd.concat(pieces, ignore_index=True)
    del pieces
    for c in frame.select_dtypes(include='object'):
        frame[c] = frame[c].astype('category')
    print(name, frame.shape, f'{frame.memory_usage(deep=True).sum()/2**20:.1f} MiB')
    return frame

train = read_table('application_train')
test = read_table('application_test')
assert train.SK_ID_CURR.is_unique and test.SK_ID_CURR.is_unique
assert not train.SK_ID_CURR.isin(test.SK_ID_CURR).any()
assert train.TARGET.notna().all() and set(train.TARGET.unique()) == {0,1}
assert 'TARGET' not in test
assert set(train.columns) - {'TARGET'} == set(test.columns)
if MODE == 'quick':
    if len(train) > QUICK_TRAIN:
        keep, _ = train_test_split(np.arange(len(train)), train_size=QUICK_TRAIN,
                                   stratify=train.TARGET, random_state=SEED)
        train = train.iloc[np.sort(keep)].reset_index(drop=True)
    test = test.sample(min(QUICK_TEST, len(test)), random_state=SEED).sort_index().reset_index(drop=True)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
all_ids = pd.Index(pd.concat([train.SK_ID_CURR, test.SK_ID_CURR], ignore_index=True))
if MODE == 'full' and len(train) > 2 * QUICK_TRAIN:
    legacy_quick, _ = train_test_split(np.arange(len(train)), train_size=QUICK_TRAIN,
                                      stratify=train.TARGET, random_state=SEED)
    candidates = np.setdiff1d(np.arange(len(train)), legacy_quick)
    _, hold_idx = train_test_split(candidates, test_size=int(.2*len(train)),
                                   stratify=train.iloc[candidates].TARGET, random_state=SEED+101)
    dev_idx = np.setdiff1d(np.arange(len(train)),hold_idx)
else:
    dev_idx,hold_idx = train_test_split(np.arange(len(train)),test_size=.2,
                                       stratify=train.TARGET,random_state=SEED)
dev_idx,hold_idx=np.sort(dev_idx),np.sort(hold_idx)
eda_labels = train.iloc[dev_idx][['SK_ID_CURR','TARGET']]
with csv_stream('HomeCredit_columns_description') as f:
    description = pd.read_csv(f, encoding='cp1252')
display(description.head())
display(train.TARGET.value_counts().to_frame('count'))
source_files = [SOURCE] if SOURCE.is_file() else sorted(MEMBERS.values())
source_fingerprint = [(str(p.resolve()),p.stat().st_size,p.stat().st_mtime_ns) for p in source_files]
cache_key = hashlib.sha256(json.dumps(dict(version=FEATURE_VERSION,source=source_fingerprint,
    mode=MODE,train_ids=train.SK_ID_CURR.tolist(),test_ids=test.SK_ID_CURR.tolist(),
    versions=versions),sort_keys=True).encode()).hexdigest()[:20]
CACHE_DIR=OUTPUT_DIR/'cache'; CACHE_DIR.mkdir(exist_ok=True)
FEATURE_CACHE=CACHE_DIR/f'features_{cache_key}.pkl'
CACHE_READY=REUSE_FEATURES and FEATURE_CACHE.exists()
print('Кэш витрины:', 'найден' if CACHE_READY else 'будет построен')


audit_rows=[]
def ratio(a, b):
    return a.div(b.where(b.ne(0))).replace([np.inf, -np.inf], np.nan)

def positive_flag(s):
    return s.gt(0).astype('float32').where(s.notna())

def compact_features(df):
    for c in df.select_dtypes(include='float64'):
        df[c] = df[c].astype('float32')
    assert df.columns.is_unique
    return df

def aggregate(df, key, prefix, specs, categories=()):
    g = df.groupby(key, observed=True, sort=False)
    out = g.size().rename(prefix + '__N_ROWS').to_frame()
    specs = {c: fs for c, fs in specs.items() if c in df}
    if specs:
        a = g.agg(specs)
        for c, fs in specs.items():
            if 'sum' in fs:
                a[c, 'sum'] = g[c].sum(min_count=1)
        a.columns = [prefix + '__' + c + '__' + f for c, f in a.columns]
        out = out.join(a)
    for c in categories:
        if c not in df:
            continue
        cat = df[c].astype('string').fillna('__MISSING__')
        counts = pd.crosstab(df[key], cat)
        counts.columns = [prefix + '__' + c + '__' + str(x) + '__SHARE' for x in counts.columns]
        counts = counts.div(g.size(), axis=0)
        out = out.join(counts)
    out.index.name = key
    assert out.index.is_unique
    return compact_features(out)

def slopes_by_loan(df, key, time_col, value_cols, min_obs=3):
    out = pd.DataFrame(index=pd.Index(df[key].unique(), name=key))
    for c in value_cols:
        if c not in df:
            continue
        q = df[[key, time_col, c]].dropna().copy()
        q['xx'] = q[time_col] ** 2
        q['xy'] = q[time_col] * q[c]
        g = q.groupby(key, observed=True)
        z = g.agg(n=(c, 'size'), sx=(time_col, 'sum'), sy=(c, 'sum'), sxx=('xx', 'sum'), sxy=('xy', 'sum'))
        slope = ratio(z.n * z.sxy - z.sx * z.sy, z.n * z.sxx - z.sx * z.sx).where(z.n >= min_obs)
        out[c + '__SLOPE'] = slope
    return out

def profile(df,name,natural_key=None):
    row=dict(table=name,rows=len(df),columns=df.shape[1])
    audit_rows.append(row)
    print('TABLE',row,flush=True)

def client_report(*args,**kwargs): pass


print('START_CELL_7',round(time.time()-START,1),flush=True)
def application_features(df):
    x = df.drop(columns=['TARGET'], errors='ignore').set_index('SK_ID_CURR').copy()
    x['EMPLOYED_SENTINEL'] = x.DAYS_EMPLOYED.eq(365243).astype('int8')
    x['DAYS_EMPLOYED'] = x.DAYS_EMPLOYED.replace(365243, np.nan)
    x['AGE_YEARS'] = -x.DAYS_BIRTH/365.25
    x['EMPLOYED_YEARS'] = -x.DAYS_EMPLOYED/365.25
    for new,a,b in [
        ('CREDIT_INCOME_RATIO','AMT_CREDIT','AMT_INCOME_TOTAL'),
        ('ANNUITY_INCOME_RATIO','AMT_ANNUITY','AMT_INCOME_TOTAL'),
        ('CREDIT_ANNUITY_RATIO','AMT_CREDIT','AMT_ANNUITY'),
        ('GOODS_CREDIT_RATIO','AMT_GOODS_PRICE','AMT_CREDIT'),
        ('EMPLOYED_AGE_RATIO','DAYS_EMPLOYED','DAYS_BIRTH'),
        ('INCOME_PER_PERSON','AMT_INCOME_TOTAL','CNT_FAM_MEMBERS')]:
        if a in x and b in x: x[new] = ratio(x[a],x[b])
    ext = [c for c in x if c.startswith('EXT_SOURCE_')]
    if ext:
        x['EXT_MEAN']=x[ext].mean(axis=1); x['EXT_MIN']=x[ext].min(axis=1)
        x['EXT_MAX']=x[ext].max(axis=1); x['EXT_STD']=x[ext].std(axis=1)
        x['EXT_COUNT']=x[ext].count(axis=1)
    docs = [c for c in x if c.startswith('FLAG_DOCUMENT_')]
    if docs: x['DOCUMENT_COUNT']=x[docs].sum(axis=1,min_count=1)
    x['MISSING_COUNT']=df.drop(columns=['TARGET','SK_ID_CURR'],errors='ignore').isna().sum(axis=1).to_numpy()
    return compact_features(x)

if not CACHE_READY:
    base = pd.concat([application_features(train),application_features(test)])
    blocks = {}
    extra=pd.DataFrame(index=base.index)
    for a,b in [('EXT_SOURCE_1','EXT_SOURCE_2'),('EXT_SOURCE_2','EXT_SOURCE_3'),('EXT_SOURCE_1','EXT_SOURCE_3')]:
        if a in base and b in base:
            extra[f'EXTRA_APP__{a}_{b}_PRODUCT']=base[a]*base[b]
            extra[f'EXTRA_APP__{a}_{b}_GAP']=(base[a]-base[b]).abs()
    extra['EXTRA_APP__EXT_AGE']=base.EXT_MEAN*base.AGE_YEARS
    extra['EXTRA_APP__EXT_EMPLOYED']=base.EXT_MEAN*base.EMPLOYED_YEARS
    extra['EXTRA_APP__INCOME_CREDIT_DIFF']=base.AMT_INCOME_TOTAL-base.AMT_CREDIT
    extra['EXTRA_APP__GOODS_CREDIT_DIFF']=base.AMT_GOODS_PRICE-base.AMT_CREDIT
    if 'OWN_CAR_AGE' in base:
        extra['EXTRA_APP__CAR_AGE_RATIO']=ratio(base.OWN_CAR_AGE,base.AGE_YEARS)
    for c in ['DAYS_REGISTRATION','DAYS_ID_PUBLISH','DAYS_LAST_PHONE_CHANGE']:
        if c in base: extra['EXTRA_APP__'+c+'_AGE_RATIO']=ratio(base[c],base.DAYS_BIRTH)
    for c in ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']:
        extra['EXTRA_APP__'+c+'_MISSING']=base[c].isna().astype('int8')
    blocks['EXTRA_APP']=compact_features(extra)


print('START_CELL_9',round(time.time()-START,1),flush=True)
if not CACHE_READY:
    bureau = read_table('bureau', all_ids)
    profile(bureau,'bureau',['SK_ID_BUREAU'])
    assert bureau.SK_ID_BUREAU.is_unique
    # Для orphan-аудита нужен полный список bureau-ID, а не quick-подмножество.
    with csv_stream('bureau') as f:
        known_bureau_ids = pd.Index(pd.read_csv(f,usecols=['SK_ID_BUREAU']).SK_ID_BUREAU)
    parts=[]; orphan_bb=0; total_bb=0
    with csv_stream('bureau_balance') as f:
        for p in pd.read_csv(f,chunksize=CHUNK_SIZE,dtype={'STATUS':'string'}):
            total_bb += len(p)
            orphan_bb += int((~p.SK_ID_BUREAU.isin(known_bureau_ids)).sum())
            parts.append(p.loc[p.SK_ID_BUREAU.isin(bureau.SK_ID_BUREAU)].copy())
    bb=pd.concat(parts,ignore_index=True); del parts, known_bureau_ids
    bb['STATUS']=bb.STATUS.astype('category')
    print('Строк bureau_balance без ключа в полном bureau:', orphan_bb, '/',total_bb)
    profile(bb,'bureau_balance',['SK_ID_BUREAU','MONTHS_BALANCE'])
    assert not bb.duplicated(['SK_ID_BUREAU','MONTHS_BALANCE']).any(), 'Разберите дубли кредит-месяц до расчёта'
    assert bb.MONTHS_BALANCE.le(0).all()
    status_num = pd.to_numeric(bb.STATUS.astype('string'),errors='coerce').astype('float64')
    bb['DELINQUENT']=positive_flag(status_num)
    bb['SEVERITY']=status_num
    bb['DELINQUENT_MONTH']=bb.MONTHS_BALANCE.where(bb.DELINQUENT.eq(1))
    bb_spec={'MONTHS_BALANCE':['min','max'], 'DELINQUENT':['mean','max','sum'],
             'SEVERITY':['mean','max'], 'DELINQUENT_MONTH':['max']}
    bb_loan=aggregate(bb,'SK_ID_BUREAU','BB',bb_spec,['STATUS'])
    bb_recent=aggregate(bb.loc[bb.MONTHS_BALANCE.between(-11,0)],'SK_ID_BUREAU','BB_12M',bb_spec)
    bb_loan=bb_loan.join(bb_recent)
    latest=bb.sort_values(['SK_ID_BUREAU','MONTHS_BALANCE']).drop_duplicates('SK_ID_BUREAU',keep='last')
    bb_loan=bb_loan.join(latest.set_index('SK_ID_BUREAU')[['SEVERITY']].rename(columns={'SEVERITY':'BB_LAST_SEVERITY'}))
    bb_mapped=bb_loan.join(bureau.set_index('SK_ID_BUREAU')[['SK_ID_CURR']],how='inner')
    blocks['BB']=aggregate(bb_mapped,'SK_ID_CURR','BB_CLIENT',
                           {c:['mean','max'] for c in bb_loan.columns})
    client_report(blocks['BB'],'bb',['BB_CLIENT__BB__DELINQUENT__mean__mean',
                                      'BB_CLIENT__BB__SEVERITY__max__max'])
    # Серии считаются только по последовательным календарным месяцам.
    seq=bb.sort_values(['SK_ID_BUREAU','MONTHS_BALANCE']).copy()
    bad=seq.DELINQUENT.eq(1)
    breaks=seq.SK_ID_BUREAU.ne(seq.SK_ID_BUREAU.shift()) | seq.MONTHS_BALANCE.diff().ne(1) | ~bad
    seq['BAD_RUN']=bad.groupby(breaks.cumsum()).cumsum().astype('float32')
    seq['BAD_MONTH']=seq.MONTHS_BALANCE.where(bad)
    seq['KNOWN_STATUS']=seq.SEVERITY.notna().astype('int8')
    e=aggregate(seq,'SK_ID_BUREAU','EXTRA_BB_LOAN',{'BAD_RUN':['max'],'BAD_MONTH':['max'],'KNOWN_STATUS':['mean']})
    e=e.join(bureau.set_index('SK_ID_BUREAU')[['SK_ID_CURR']],validate='one_to_one')
    blocks['EXTRA_BB']=aggregate(e,'SK_ID_CURR','EXTRA_BB',{c:['mean','max'] for c in e if c!='SK_ID_CURR'})
    del seq,bad,breaks,e
    del bb,bb_loan,bb_recent,bb_mapped,latest,status_num; gc.collect()
    
    bureau['DEBT_RATIO']=ratio(bureau.AMT_CREDIT_SUM_DEBT,bureau.AMT_CREDIT_SUM)
    bureau['OVERDUE_RATIO']=ratio(bureau.AMT_CREDIT_SUM_OVERDUE,bureau.AMT_CREDIT_SUM)
    bureau['HAS_OVERDUE']=positive_flag(bureau.AMT_CREDIT_SUM_OVERDUE)
    bureau['DURATION']=bureau.DAYS_CREDIT_ENDDATE-bureau.DAYS_CREDIT
    bcols=['DAYS_CREDIT','DAYS_CREDIT_ENDDATE','DAYS_ENDDATE_FACT','DAYS_CREDIT_UPDATE',
           'CREDIT_DAY_OVERDUE','AMT_CREDIT_SUM','AMT_CREDIT_SUM_DEBT','AMT_CREDIT_SUM_OVERDUE',
           'AMT_CREDIT_MAX_OVERDUE','AMT_CREDIT_SUM_LIMIT','AMT_ANNUITY',
           'CNT_CREDIT_PROLONG','DEBT_RATIO','OVERDUE_RATIO','HAS_OVERDUE','DURATION']
    b_spec={c:['mean','min','max'] for c in bcols}
    for c in ['AMT_CREDIT_SUM','AMT_CREDIT_SUM_DEBT','AMT_CREDIT_SUM_OVERDUE']: b_spec[c].append('sum')
    b = aggregate(bureau,'SK_ID_CURR','BUREAU',b_spec,['CREDIT_ACTIVE','CREDIT_TYPE'])
    for label,mask in [('ACTIVE',bureau.CREDIT_ACTIVE.eq('Active')),
                       ('RECENT_1Y',bureau.DAYS_CREDIT.between(-365,0))]:
        b=b.join(aggregate(bureau.loc[mask],'SK_ID_CURR','BUREAU_'+label,b_spec))
    b['BUREAU__TOTAL_DEBT_RATIO']=ratio(b['BUREAU__AMT_CREDIT_SUM_DEBT__sum'],b['BUREAU__AMT_CREDIT_SUM__sum'])
    blocks['BUREAU']=b
    client_report(b,'bureau',['BUREAU__N_ROWS','BUREAU__TOTAL_DEBT_RATIO','BUREAU__HAS_OVERDUE__mean'])
    # Сроки/типы внешних кредитов и интервалы между открытиями.
    sorted_b=bureau.sort_values(['SK_ID_CURR','DAYS_CREDIT','SK_ID_BUREAU']).copy()
    sorted_b['OPENING_GAP']=sorted_b.groupby('SK_ID_CURR').DAYS_CREDIT.diff()
    sorted_b['ACTIVE']=sorted_b.CREDIT_ACTIVE.eq('Active').astype('int8')
    sorted_b['ENDS_FUTURE']=positive_flag(sorted_b.DAYS_CREDIT_ENDDATE)
    blocks['EXTRA_BUREAU']=aggregate(sorted_b,'SK_ID_CURR','EXTRA_BUREAU',
      {'OPENING_GAP':['mean','min','max'],'ACTIVE':['sum','mean'],'ENDS_FUTURE':['mean']})
    del sorted_b,bureau,b; gc.collect()

print('START_CELL_11',round(time.time()-START,1),flush=True)
if not CACHE_READY:
    prev=read_table('previous_application',all_ids)
    profile(prev,'previous_application',['SK_ID_PREV'])
    assert prev.SK_ID_PREV.is_unique
    assert prev.DAYS_DECISION.dropna().le(0).all()
    for c in [c for c in prev if c.startswith('DAYS_')]:
        prev[c+'_SENTINEL']=prev[c].eq(365243).astype('int8')
        prev[c]=prev[c].replace(365243,np.nan)
    prev['APPLICATION_CREDIT_RATIO']=ratio(prev.AMT_APPLICATION,prev.AMT_CREDIT)
    prev['ANNUITY_CREDIT_RATIO']=ratio(prev.AMT_ANNUITY,prev.AMT_CREDIT)
    prev['DOWNPAYMENT_CREDIT_RATIO']=ratio(prev.AMT_DOWN_PAYMENT,prev.AMT_CREDIT)
    pcols=['AMT_ANNUITY','AMT_APPLICATION','AMT_CREDIT','AMT_DOWN_PAYMENT','AMT_GOODS_PRICE',
           'HOUR_APPR_PROCESS_START','RATE_DOWN_PAYMENT','DAYS_DECISION','CNT_PAYMENT',
           'APPLICATION_CREDIT_RATIO','ANNUITY_CREDIT_RATIO','DOWNPAYMENT_CREDIT_RATIO']
    pcols += [c for c in prev if c.startswith('DAYS_') and c!='DAYS_DECISION']
    p_spec={c:['mean','min','max'] for c in pcols}
    p=aggregate(prev,'SK_ID_CURR','PREV',p_spec,
                ['NAME_CONTRACT_STATUS','NAME_CONTRACT_TYPE','NAME_CLIENT_TYPE','NAME_YIELD_GROUP','CHANNEL_TYPE'])
    for label,mask in [('APPROVED',prev.NAME_CONTRACT_STATUS.eq('Approved')),
                       ('REFUSED',prev.NAME_CONTRACT_STATUS.eq('Refused')),
                       ('RECENT_1Y',prev.DAYS_DECISION.between(-365,0))]:
        p=p.join(aggregate(prev.loc[mask],'SK_ID_CURR','PREV_'+label,p_spec))
    last=prev.sort_values(['SK_ID_CURR','DAYS_DECISION','SK_ID_PREV']).drop_duplicates('SK_ID_CURR',keep='last')
    last=last.set_index('SK_ID_CURR')[['NAME_CONTRACT_STATUS','AMT_CREDIT','DAYS_DECISION']].add_prefix('PREV_LAST__')
    p=p.join(last)
    blocks['PREV']=p
    client_report(p,'previous_application',['PREV__N_ROWS','PREV__APPLICATION_CREDIT_RATIO__mean',
     'PREV__NAME_CONTRACT_STATUS__Refused__SHARE','PREV__DAYS_DECISION__max'])
    prev_sorted=prev.sort_values(['SK_ID_CURR','DAYS_DECISION','SK_ID_PREV']).copy()
    prev_sorted['DECISION_GAP']=prev_sorted.groupby('SK_ID_CURR').DAYS_DECISION.diff()
    prev_sorted['REFUSED']=prev_sorted.NAME_CONTRACT_STATUS.eq('Refused').astype('int8')
    prev_sorted['APPROVED']=prev_sorted.NAME_CONTRACT_STATUS.eq('Approved').astype('int8')
    prev_sorted['REFUSED_DAY']=prev_sorted.DAYS_DECISION.where(prev_sorted.REFUSED.eq(1))
    es={'REFUSED':['mean'],'APPROVED':['mean'],'DECISION_GAP':['mean','min'],
        'REFUSED_DAY':['max'],'AMT_CREDIT':['mean'],'AMT_ANNUITY':['mean']}
    e=aggregate(prev_sorted,'SK_ID_CURR','EXTRA_PREV',es,['CODE_REJECT_REASON','NAME_PRODUCT_TYPE','PRODUCT_COMBINATION'])
    for k in [3,5]:
        recent=prev_sorted.groupby('SK_ID_CURR',sort=False).tail(k)
        e=e.join(aggregate(recent,'SK_ID_CURR',f'EXTRA_PREV_LAST{k}',es))
    blocks['EXTRA_PREV']=e
    del prev_sorted,recent,e,prev,p,last; gc.collect()

print('START_CELL_13',round(time.time()-START,1),flush=True)
if not CACHE_READY:
    def monthly_features(df,prefix,cols):
        key=['SK_ID_PREV','MONTHS_BALANCE']
        assert not df.duplicated(key).any(), f'{prefix}: разберите дубли договор-месяц'
        assert df.MONTHS_BALANCE.le(0).all()
        assert df.groupby('SK_ID_PREV').SK_ID_CURR.nunique().le(1).all()
        specs={c:['mean','max'] for c in cols if c in df}
        specs['MONTHS_BALANCE']=['min','max']
        out=aggregate(df,'SK_ID_CURR',prefix+'_ROW',specs,['NAME_CONTRACT_STATUS'])
        for w in [6,12]:
            out=out.join(aggregate(df.loc[df.MONTHS_BALANCE.between(1-w,0)],
                                   'SK_ID_CURR',f'{prefix}_{w}M',specs))
        loans=aggregate(df,'SK_ID_PREV',prefix+'_LOAN',specs)
        mapping=df[['SK_ID_PREV','SK_ID_CURR']].drop_duplicates().set_index('SK_ID_PREV')
        loans=loans.join(mapping,validate='one_to_one')
        loan_cols=[c for c in loans if c!='SK_ID_CURR']
        out=out.join(aggregate(loans,'SK_ID_CURR',prefix+'_CLIENT',
                               {c:['mean','max'] for c in loan_cols}))
        latest=df.sort_values(['SK_ID_PREV','MONTHS_BALANCE']).drop_duplicates('SK_ID_PREV',keep='last')
        out=out.join(aggregate(latest,'SK_ID_CURR',prefix+'_LAST',specs))
        return out
    
    def extra_monthly(df,prefix,cols):
        recent=df.loc[df.MONTHS_BALANCE.between(-11,0)]
        slopes=slopes_by_loan(recent,'SK_ID_PREV','MONTHS_BALANCE',cols)
        mapping=df[['SK_ID_PREV','SK_ID_CURR']].drop_duplicates().set_index('SK_ID_PREV')
        slopes=slopes.join(mapping,validate='one_to_one')
        out=aggregate(slopes,'SK_ID_CURR','EXTRA_'+prefix+'_TREND',
                      {c:['mean','max'] for c in slopes if c!='SK_ID_CURR'})
        for c in cols:
            if c not in df: continue
            recent_mean=df.loc[df.MONTHS_BALANCE.between(-5,0)].groupby('SK_ID_CURR')[c].mean()
            previous_mean=df.loc[df.MONTHS_BALANCE.between(-11,-6)].groupby('SK_ID_CURR')[c].mean()
            out['EXTRA_'+prefix+'__'+c+'__RECENT_MINUS_PREVIOUS']=recent_mean-previous_mean
        if 'SK_DPD' in df:
            last_bad=df.loc[df.SK_DPD.gt(0)].groupby('SK_ID_CURR').MONTHS_BALANCE.max()
            out['EXTRA_'+prefix+'__MONTHS_SINCE_DPD']=-last_bad
        return compact_features(out)
    
    pos=read_table('POS_CASH_balance',all_ids)
    profile(pos,'POS_CASH_balance',['SK_ID_PREV','MONTHS_BALANCE'])
    pos['HAS_DPD']=positive_flag(pos.SK_DPD)
    pos['REMAINING_SHARE']=ratio(pos.CNT_INSTALMENT_FUTURE,pos.CNT_INSTALMENT)
    blocks['POS']=monthly_features(pos,'POS',['SK_DPD','SK_DPD_DEF','HAS_DPD',
                                            'CNT_INSTALMENT','CNT_INSTALMENT_FUTURE','REMAINING_SHARE'])
    client_report(blocks['POS'],'pos',['POS_ROW__HAS_DPD__mean','POS_LAST__CNT_INSTALMENT_FUTURE__mean'])
    blocks['EXTRA_POS']=extra_monthly(pos,'POS',['SK_DPD','REMAINING_SHARE'])
    del pos; gc.collect()
    
    cc=read_table('credit_card_balance',all_ids)
    profile(cc,'credit_card_balance',['SK_ID_PREV','MONTHS_BALANCE'])
    cc['UTILIZATION']=ratio(cc.AMT_BALANCE,cc.AMT_CREDIT_LIMIT_ACTUAL)
    cc['PAYMENT_MIN_RATIO']=ratio(cc.AMT_PAYMENT_TOTAL_CURRENT,cc.AMT_INST_MIN_REGULARITY)
    cc['CASH_SHARE']=ratio(cc.AMT_DRAWINGS_ATM_CURRENT,cc.AMT_DRAWINGS_CURRENT)
    cc['HAS_DPD']=positive_flag(cc.SK_DPD)
    cc_cols=['AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL','AMT_DRAWINGS_CURRENT','AMT_DRAWINGS_ATM_CURRENT',
             'AMT_PAYMENT_TOTAL_CURRENT','AMT_INST_MIN_REGULARITY','CNT_DRAWINGS_CURRENT',
             'SK_DPD','SK_DPD_DEF','UTILIZATION','PAYMENT_MIN_RATIO','CASH_SHARE','HAS_DPD']
    blocks['CC']=monthly_features(cc,'CC',cc_cols)
    client_report(blocks['CC'],'credit_card',['CC_ROW__UTILIZATION__mean','CC_ROW__HAS_DPD__mean',
                                           'CC_ROW__PAYMENT_MIN_RATIO__mean'])
    blocks['EXTRA_CC']=extra_monthly(cc,'CC',['UTILIZATION','PAYMENT_MIN_RATIO','AMT_BALANCE','SK_DPD'])
    del cc; gc.collect()

print('START_CELL_15',round(time.time()-START,1),flush=True)
if not CACHE_READY:
    def installment_level(raw):
        k=['SK_ID_CURR','SK_ID_PREV','NUM_INSTALMENT_VERSION','NUM_INSTALMENT_NUMBER']
        r=raw.copy()
        r['PAID_OBS']=r.AMT_PAYMENT.where(r.DAYS_ENTRY_PAYMENT.le(0))
        # Для известного платежа после срока вклад к сроку = 0.
        r['PAID_ON_TIME']=r['PAID_OBS'].where(r.DAYS_ENTRY_PAYMENT.le(r.DAYS_INSTALMENT),0)
        r.loc[r.DAYS_ENTRY_PAYMENT.isna() | r.AMT_PAYMENT.isna(),'PAID_ON_TIME']=np.nan
        r['UNKNOWN_PAYMENT']=(r.AMT_PAYMENT.isna() | r.DAYS_ENTRY_PAYMENT.isna()).astype('int8')
        g=r.groupby(k,sort=False,observed=True)
        scheduled=g.agg(DUE=('DAYS_INSTALMENT','first'),EXPECTED=('AMT_INSTALMENT','first'),
                        DUE_UNIQUE=('DAYS_INSTALMENT','nunique'),AMOUNT_UNIQUE=('AMT_INSTALMENT','nunique'),
                        PARTS=('AMT_PAYMENT','size'),UNKNOWN_PARTS=('UNKNOWN_PAYMENT','sum'))
        scheduled['PAID']=g.PAID_OBS.sum(min_count=1)
        scheduled['PAID_ON_TIME']=g.PAID_ON_TIME.sum(min_count=1)
        # При неполном наблюдении суммы/даты консервативно не выводим факт недоплаты.
        unknown=scheduled.UNKNOWN_PARTS.gt(0)
        scheduled.loc[unknown,['PAID','PAID_ON_TIME']]=np.nan
        conflict=scheduled.DUE_UNIQUE.ne(1) | scheduled.AMOUNT_UNIQUE.ne(1)
        conflicts=scheduled.loc[conflict].reset_index()
        s=scheduled.loc[~conflict & scheduled.DUE.le(0)].copy()
        tol=.01  # фиксированная погрешность денежной суммы
        paid=r.loc[r.DAYS_ENTRY_PAYMENT.le(0) & r.AMT_PAYMENT.notna()].copy()
        paid=paid.sort_values(k+['DAYS_ENTRY_PAYMENT'])
        paid['CUM_PAID']=paid.groupby(k,sort=False,observed=True).AMT_PAYMENT.cumsum()
        # Первая дата, когда накопленная выплата достигает величины взноса.
        reached=paid.loc[paid.CUM_PAID.ge(paid.AMT_INSTALMENT-tol)]
        completion=reached.groupby(k,observed=True).DAYS_ENTRY_PAYMENT.min()
        s=s.join(completion.rename('COMPLETED_DAY'))
        s['PAYMENT_RATIO']=ratio(s.PAID,s.EXPECTED)
        s['SHORTFALL']=(s.EXPECTED-s.PAID).clip(lower=0)
        s['UNDERPAID']=s.PAID.lt(s.EXPECTED-tol).astype('float32').where(s.PAID.notna())
        s['NOT_FULL_BY_DUE']=s.PAID_ON_TIME.lt(s.EXPECTED-tol).astype('float32').where(s.PAID_ON_TIME.notna())
        s['COMPLETED_DAY']=s.COMPLETED_DAY.where(s.PAID.ge(s.EXPECTED-tol))
        s['DAYS_LATE_COMPLETION']=(s.COMPLETED_DAY-s.DUE).clip(lower=0)
        s['DAYS_SINCE_DUE_IF_UNDERPAID']=(-s.DUE).where(s.UNDERPAID.eq(1))
        return s.reset_index(),conflicts
    
    ins=read_table('installments_payments',all_ids)
    profile(ins,'installments_payments',
            ['SK_ID_PREV','NUM_INSTALMENT_VERSION','NUM_INSTALMENT_NUMBER'])
    print('Платежи с датой после заявки:',int(ins.DAYS_ENTRY_PAYMENT.gt(0).sum()))
    inst,conflicts=installment_level(ins)
    print('Конфликтующих взносов исключено:',len(conflicts))
    conflicts.to_csv(REPORT_DIR/'installment_schedule_conflicts.csv',index=False)
    del ins; gc.collect()
    i_spec={c:['mean','max'] for c in ['PAYMENT_RATIO','SHORTFALL','UNDERPAID','NOT_FULL_BY_DUE',
              'DAYS_LATE_COMPLETION','DAYS_SINCE_DUE_IF_UNDERPAID','PARTS','UNKNOWN_PARTS']}
    i_spec.update({'DUE':['min','max'],'EXPECTED':['sum','mean'],'PAID':['sum','mean']})
    i=aggregate(inst,'SK_ID_CURR','INST',i_spec)
    for days in [180,365]:
        i=i.join(aggregate(inst.loc[inst.DUE.between(-days,0)],'SK_ID_CURR',f'INST_{days}D',i_spec))
    loan=aggregate(inst,'SK_ID_PREV','INST_LOAN',i_spec)
    mapping=inst[['SK_ID_PREV','SK_ID_CURR']].drop_duplicates().set_index('SK_ID_PREV')
    assert mapping.index.is_unique
    loan=loan.join(mapping,validate='one_to_one')
    i=i.join(aggregate(loan,'SK_ID_CURR','INST_CLIENT',{c:['mean','max'] for c in loan if c!='SK_ID_CURR'}))
    blocks['INST']=i
    client_report(i,'installments',['INST__NOT_FULL_BY_DUE__mean','INST__UNDERPAID__mean',
                                   'INST__DAYS_LATE_COMPLETION__max'])
    inst['ON_TIME_RATIO']=ratio(inst.PAID_ON_TIME,inst.EXPECTED)
    inst['SHORTFALL_BY_DUE']=(inst.EXPECTED-inst.PAID_ON_TIME).clip(lower=0)
    inst['LATE_7']=inst.DAYS_LATE_COMPLETION.gt(7).astype('float32').where(inst.DAYS_LATE_COMPLETION.notna())
    inst['LATE_30']=inst.DAYS_LATE_COMPLETION.gt(30).astype('float32').where(inst.DAYS_LATE_COMPLETION.notna())
    inst['EARLY_DAYS']=(inst.DUE-inst.COMPLETED_DAY).clip(lower=0)
    inst['LATE_DUE']=inst.DUE.where(inst.NOT_FULL_BY_DUE.eq(1))
    es={c:['mean','max'] for c in ['ON_TIME_RATIO','SHORTFALL_BY_DUE','LATE_7','LATE_30','EARLY_DAYS','NOT_FULL_BY_DUE','DAYS_LATE_COMPLETION']}
    es['LATE_DUE']=['max']
    e=aggregate(inst,'SK_ID_CURR','EXTRA_INST',es)
    ordered=inst.sort_values(['SK_ID_CURR','DUE','SK_ID_PREV','NUM_INSTALMENT_NUMBER'])
    for k in [3,10]:
        e=e.join(aggregate(ordered.groupby('SK_ID_CURR').tail(k),'SK_ID_CURR',f'EXTRA_INST_LAST{k}',es))
    # Denominator uses only rows with known feature values.
    w=np.exp(inst.DUE/180.)
    for c in ['NOT_FULL_BY_DUE','DAYS_LATE_COMPLETION','ON_TIME_RATIO']:
        numerator=(inst[c]*w).groupby(inst.SK_ID_CURR).sum(min_count=1)
        denominator=w.where(inst[c].notna()).groupby(inst.SK_ID_CURR).sum(min_count=1)
        e['EXTRA_INST__'+c+'__DECAY_MEAN']=ratio(numerator,denominator)
    blocks['EXTRA_INST']=compact_features(e)
    del ordered,e,w,inst,conflicts,i,loan,mapping; gc.collect()

print('START_CELL_17',round(time.time()-START,1),flush=True)
if not CACHE_READY:
    features=base.copy()
    block_columns={}
    for name,block in blocks.items():
        assert block.index.is_unique and block.index.name=='SK_ID_CURR'
        assert not set(block.columns)&set(features.columns)
        block=block.copy()
        flag=f'{name}__HAS_HISTORY'
        block[flag]=1
        features=features.join(block,how='left',validate='one_to_one')
        features[flag]=features[flag].fillna(0).astype('int8')
        # Только прямой размер блока, а не все вложенные *_N_ROWS агрегаты.
        direct=[c for c in block if c.endswith('__N_ROWS')]
        for c in direct: features[c]=features[c].fillna(0)
        block_columns[name]=list(block.columns)
    # Текущая заявка относительно собственных исторических кредитов.
    cross=pd.DataFrame(index=features.index)
    for new,a,b in [
     ('EXTRA_CROSS__BUREAU_DEBT_INCOME','BUREAU__AMT_CREDIT_SUM_DEBT__sum','AMT_INCOME_TOTAL'),
     ('EXTRA_CROSS__CURRENT_PREV_CREDIT','AMT_CREDIT','PREV_APPROVED__AMT_CREDIT__mean'),
     ('EXTRA_CROSS__CURRENT_PREV_ANNUITY','AMT_ANNUITY','PREV_APPROVED__AMT_ANNUITY__mean'),
     ('EXTRA_CROSS__BUREAU_OVERDUE_INCOME','BUREAU__AMT_CREDIT_SUM_OVERDUE__sum','AMT_INCOME_TOTAL')]:
        if a in features and b in features: cross[new]=ratio(features[a],features[b])
    features=features.join(compact_features(cross),validate='one_to_one')
    block_columns['EXTRA_CROSS']=list(cross.columns)
    assert len(features)==len(train)+len(test)
    assert features.index.is_unique and features.columns.is_unique
    assert not any(c in features for c in ['TARGET','SK_ID_CURR','SK_ID_PREV','SK_ID_BUREAU'])
    num=features.select_dtypes(include=np.number).columns
    features[num]=features[num].replace([np.inf,-np.inf],np.nan)
    features=compact_features(features)
    X=features.loc[train.SK_ID_CURR].reset_index(drop=True)
    X_test=features.loc[test.SK_ID_CURR].reset_index(drop=True)
    y=train.TARGET.astype('int8').reset_index(drop=True)
    # Безопасные уникальные названия для LightGBM. Семантический словарь сохраняется.
    name_map={c:f'f{j:04d}_'+re.sub('[^A-Za-z0-9_]+','_',c) for j,c in enumerate(X.columns)}
    original_columns=list(X.columns)
    X=X.rename(columns=name_map); X_test=X_test.rename(columns=name_map)
    base_columns=[name_map[c] for c in base.columns]
    block_columns={k:[name_map[c] for c in v] for k,v in block_columns.items()}
    pd.DataFrame({'original':list(name_map),'model':list(name_map.values())}).to_csv(OUTPUT_DIR/'feature_dictionary.csv',index=False)
    pd.DataFrame(audit_rows).to_csv(REPORT_DIR/'table_audit.csv',index=False)
    print('Матрица:',X.shape,'test:',X_test.shape,'memory MiB:',X.memory_usage(deep=True).sum()/2**20)
    if SAVE_FEATURES:
        X.assign(SK_ID_CURR=train.SK_ID_CURR,TARGET=y).to_csv(OUTPUT_DIR/'train_features.csv.gz',index=False)
        X_test.assign(SK_ID_CURR=test.SK_ID_CURR).to_csv(OUTPUT_DIR/'test_features.csv.gz',index=False)
    del features,blocks,base,block,cross; gc.collect()
    legacy_columns=[c for c in X if not c.split('_',1)[1].startswith('EXTRA_')]
    if SAVE_FEATURE_CACHE:
        pd.to_pickle(dict(X=X,X_test=X_test,y=y,name_map=name_map,base_columns=base_columns,
                         block_columns=block_columns,legacy_columns=legacy_columns),FEATURE_CACHE)
else:
    # Только кэш, созданный этим notebook из ваших файлов.
    state=pd.read_pickle(FEATURE_CACHE)
    X,X_test,y=(state[k] for k in ['X','X_test','y'])
    name_map,base_columns,block_columns,legacy_columns=(state[k] for k in ['name_map','base_columns','block_columns','legacy_columns'])
    del state
    print('Витрина восстановлена:',X.shape,X_test.shape)
assert len(X)==len(train) and len(X_test)==len(test)
# Feature/row hash prevents reuse of predictions on different values or row order.
matrix_hash=hashlib.sha256(pd.util.hash_pandas_object(X,index=True).values.tobytes()+
    y.to_numpy().tobytes()+train.SK_ID_CURR.to_numpy().tobytes()).hexdigest()
discovery_idx,compare_idx=train_test_split(dev_idx,train_size=.3,stratify=y.iloc[dev_idx],random_state=SEED+11)
discovery_idx,compare_idx=np.sort(discovery_idx),np.sort(compare_idx)
folds=list(StratifiedKFold(N_SPLITS,shuffle=True,random_state=SEED).split(compare_idx,y.iloc[compare_idx]))
split_table=pd.DataFrame({'SK_ID_CURR':train.SK_ID_CURR,'split':'holdout','fold':-1})
split_table.loc[discovery_idx,'split']='discovery'
split_table.loc[compare_idx,'split']='comparison'
for f,(_,va) in enumerate(folds):split_table.loc[compare_idx[va],'fold']=f
split_table.to_csv(OUTPUT_DIR/'splits.csv',index=False)
print(split_table.groupby('split').size())
print('v1 features:',len(legacy_columns),'v2 features:',X.shape[1])



X.to_parquet(OUTPUT_DIR/'train_features.parquet',index=False,compression='zstd')
X_test.to_parquet(OUTPUT_DIR/'test_features.parquet',index=False,compression='zstd')
train[['SK_ID_CURR','TARGET']].to_parquet(OUTPUT_DIR/'train_labels.parquet',index=False)
test[['SK_ID_CURR']].to_parquet(OUTPUT_DIR/'test_ids.parquet',index=False)
with csv_stream('sample_submission') as f: pd.read_csv(f).to_csv(OUTPUT_DIR/'sample_submission.csv',index=False)
np.savez_compressed(OUTPUT_DIR/'original_splits.npz',dev_idx=dev_idx,hold_idx=hold_idx,discovery_idx=discovery_idx,compare_idx=compare_idx)
manifest=dict(feature_version=FEATURE_VERSION,versions=versions,train_rows=len(X),test_rows=len(X_test),features=X.shape[1],matrix_hash=matrix_hash,name_map=name_map,base_columns=base_columns,block_columns=block_columns,legacy_columns=legacy_columns,seconds=time.time()-START,source_fingerprint=source_fingerprint)
(OUTPUT_DIR/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('FEATURE_CACHE_COMPLETE',X.shape,X_test.shape,'seconds',round(time.time()-START,1),flush=True)
