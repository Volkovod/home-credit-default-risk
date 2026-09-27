"""Reproducible GPU benchmark, development CV, frozen holdout, full-data refit."""
from pathlib import Path
import concurrent.futures, gc, hashlib, importlib.metadata, itertools, json, os, subprocess, time
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

START = time.time()
SEED = 42
CV_FOLDS = 5
MAX_TREES = 4000
PATIENCE = 150
OUT = Path('/kaggle/working/home_credit_research')
OUT.mkdir(exist_ok=True)
CACHE = next(p.parent for p in Path('/kaggle/input').rglob('manifest.json')
             if json.loads(p.read_text()).get('feature_version')=='research-v2-20260926')
manifest = json.loads((CACHE/'manifest.json').read_text())
X = pd.read_parquet(CACHE/'train_features.parquet')
XT = pd.read_parquet(CACHE/'test_features.parquet')
labels = pd.read_parquet(CACHE/'train_labels.parquet')
test_ids = pd.read_parquet(CACHE/'test_ids.parquet')
y = labels.TARGET.astype('int8')
with np.load(CACHE/'original_splits.npz') as data:
    dev = data['dev_idx']; hold = data['hold_idx']
assert len(X)==len(y)==307511 and len(XT)==len(test_ids)==48744
assert X.columns.equals(XT.columns) and labels.SK_ID_CURR.is_unique and test_ids.SK_ID_CURR.is_unique
assert set(dev).isdisjoint(hold) and len(dev)+len(hold)==len(y)
assert not set(labels.SK_ID_CURR)&set(test_ids.SK_ID_CURR)
assert not any(c in X for c in ['TARGET','SK_ID_CURR','SK_ID_PREV','SK_ID_BUREAU'])
original_cols = [c for c in X if not c.startswith('R2_')]
assert len(original_cols)==manifest['original_feature_count']
name_map = manifest['name_map']
tr, va = train_test_split(dev, test_size=.2, stratify=y.iloc[dev], random_state=SEED+2026)
tr=np.sort(tr); va=np.sort(va)
versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','lightgbm','xgboost','catboost']}
gpus=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
environment=dict(versions=versions,gpus=gpus,seed=SEED,cv_folds=CV_FOLDS,feature_version=manifest['feature_version'])
(OUT/'environment.json').write_text(json.dumps(environment,indent=2))
print('ENVIRONMENT',json.dumps(environment),flush=True)
pd.DataFrame({'SK_ID_CURR':labels.SK_ID_CURR,'split':np.where(np.isin(np.arange(len(y)),hold),'holdout','development')}).to_csv(OUT/'split_manifest.csv',index=False)

def metrics(yy, p):
    p=np.asarray(p)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    return dict(auc=float(roc_auc_score(yy,p)),ap=float(average_precision_score(yy,p)),logloss=float(log_loss(yy,p)))

def cohort_key(frame, columns):
    return frame[[name_map[c] for c in columns]].astype('string').fillna('__MISSING__').agg('|'.join,axis=1)

def prepare(train_frame, other_frames, spec):
    cols=original_cols if spec['features']=='original' else list(X.columns)
    frames=[f[cols].copy() for f in [train_frame]+list(other_frames)]
    peer_state={}
    if spec['features']=='peer':
        for group in [('NAME_EDUCATION_TYPE','CODE_GENDER'),('OCCUPATION_TYPE','CODE_GENDER'),
                      ('ORGANIZATION_TYPE',),('REGION_RATING_CLIENT','NAME_INCOME_TYPE')]:
            keys=[cohort_key(f,group) for f in frames]
            for raw in ['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','EXT_SOURCE_2','EXT_SOURCE_3']:
                col=name_map[raw]; stem='PEER_'+'_'.join(group)+'__'+raw
                # Only current training-fold features determine the reference statistics.
                stats=frames[0][col].groupby(keys[0]).agg(['median','count'])
                stats=stats.loc[stats['count']>=30]
                med=stats['median']; fallback=float(frames[0][col].median())
                peer_state[stem]=dict(group=list(group),feature=raw,fallback=fallback,medians=med.to_dict())
                for f,key in zip(frames,keys):
                    reference=key.map(med).fillna(fallback).astype('float32')
                    f[stem+'_DIFF']=(f[col]-reference).astype('float32')
                    f[stem+'_RATIO']=(f[col]/reference.where(reference.abs()>1e-9)).astype('float32')
    categories=list(frames[0].select_dtypes(include=['object','string','category']).columns)
    schema={}
    for c in categories:
        if spec['kind']=='cat':
            for frame in frames: frame[c]=frame[c].astype('string').fillna('__MISSING__').astype(str)
        else:
            cats=sorted(frames[0][c].astype('string').dropna().unique().tolist())
            schema[c]=cats; dtype=pd.CategoricalDtype(cats)
            for frame in frames: frame[c]=frame[c].astype('string').astype(dtype)
    return frames,categories,dict(categories=schema,peer=peer_state,columns=list(frames[0]))

# Regression check: validation features must never alter fitted cohort statistics.
toy_spec=dict(features='peer',kind='xgb')
toy_train=X.iloc[tr[:80]].copy(); toy_valid=X.iloc[va[:12]].copy()
prepared_a,_,schema_a=prepare(toy_train,[toy_valid],toy_spec)
changed=toy_valid.copy(); changed[name_map['AMT_INCOME_TOTAL']]=1e12
prepared_b,_,schema_b=prepare(toy_train,[changed],toy_spec)
pd.testing.assert_frame_equal(prepared_a[0],prepared_b[0])
assert schema_a==schema_b
del toy_train,toy_valid,changed,prepared_a,prepared_b,schema_a,schema_b
print('CHECK_PASSED validation cannot change training transformations',flush=True)

def fit(spec, train_ix, val_ix=None, extras=(), seed=42, trees=MAX_TREES, tag='model', device=0):
    start=time.time()
    others=([X.iloc[val_ix]] if val_ix is not None else [])+list(extras)
    frames,cats,schema=prepare(X.iloc[train_ix],others,spec)
    a=frames[0]; rest=frames[1:]; v=rest[0] if val_ix is not None else None
    params=dict(spec['params']); kind=spec['kind']; start_fit=time.time()
    if kind=='xgb':
        params.update(device=f'cuda:{device}',n_jobs=2)
        if v is not None:params['early_stopping_rounds']=PATIENCE
        model=xgb.XGBClassifier(**params,n_estimators=trees,random_state=seed)
        kwargs=dict(eval_set=[(v,y.iloc[val_ix])]) if v is not None else {}
        model.fit(a,y.iloc[train_ix],verbose=500,**kwargs)
        actual=json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
        assert actual.startswith('cuda'),actual
        iterations=int(model.best_iteration+1) if v is not None else trees
        model.save_model(OUT/f'{tag}.ubj')
    elif kind=='lgb':
        params.update(gpu_device_id=device,n_jobs=2)
        model=lgb.LGBMClassifier(**params,n_estimators=trees,random_state=seed)
        kwargs=dict(eval_set=[(v,y.iloc[val_ix])],callbacks=[lgb.early_stopping(PATIENCE,first_metric_only=True,verbose=False),lgb.log_evaluation(500)]) if v is not None else {}
        model.fit(a,y.iloc[train_ix],**kwargs)
        iterations=int(model.best_iteration_ or trees)
        model.booster_.save_model(str(OUT/f'{tag}.txt'))
    elif kind=='cat':
        params.update(devices=str(device),thread_count=2)
        model=CatBoostClassifier(**params,iterations=trees,random_seed=seed)
        kwargs=dict(eval_set=(v,y.iloc[val_ix]),early_stopping_rounds=PATIENCE,use_best_model=True) if v is not None else {}
        model.fit(a,y.iloc[train_ix],cat_features=cats,verbose=500,**kwargs)
        iterations=int(model.tree_count_); model.save_model(str(OUT/f'{tag}.cbm'))
    else:raise ValueError(kind)
    fit_seconds=time.time()-start_fit
    preds=[model.predict_proba(frame)[:,1] for frame in rest]
    for p in preds: assert np.isfinite(p).all()
    info=dict(name=spec['name'],kind=kind,features=spec['features'],iterations=iterations,
              fit_seconds=fit_seconds,total_seconds=time.time()-start,seed=seed,device=device,n_features=a.shape[1])
    (OUT/f'{tag}_metadata.json').write_text(json.dumps(dict(**info,spec=spec),indent=2))
    (OUT/f'{tag}_schema.json').write_text(json.dumps(schema))
    pd.DataFrame({'feature':a.columns,'importance':model.feature_importances_}).sort_values('importance',ascending=False).to_csv(OUT/f'{tag}_importance.csv',index=False)
    del model,frames,a,rest,v,others
    gc.collect()
    return preds,info

def weight_grid(names):
    options=[{n:1.} for n in names]
    for a,b in itertools.combinations(names,2):
        options += [{a:w,b:1-w} for w in [.25,.5,.75]]
    return options

def compare(predictions, truth):
    result=[]
    for weights in weight_grid(list(predictions)):
        p=sum(w*predictions[n] for n,w in weights.items())
        result.append(dict(weights=weights,**metrics(truth,p)))
    return sorted(result,key=lambda r:r['auc'],reverse=True)

# Reuse the completed baseline predictions only after verifying validation IDs and labels.
rows=[]; specs={}; predictions={}
for old_name in ['notebookd57678da90','home-credit-gpu-tuning-20260925']:
    matches=[p for p in Path('/kaggle/input').rglob('benchmark_predictions.csv') if old_name in str(p)]
    if not matches:continue
    source=matches[0].parent
    old_pred=pd.read_csv(source/'benchmark_predictions.csv')
    assert np.array_equal(old_pred.SK_ID_CURR,labels.iloc[va].SK_ID_CURR)
    assert np.array_equal(old_pred.TARGET,y.iloc[va])
    old_specs=json.loads((source/'benchmark_specs.json').read_text())
    old_rows=pd.read_csv(source/'benchmark.csv').to_dict('records')
    for spec in old_specs:
        if spec['name'] not in ['xgb_gpu_depth7','cat_gpu_depth6','lgb_gpu_127_slow']:continue
        spec={**spec,'features':'original'}; name=spec['name']; specs[name]=spec
        p=old_pred[name].to_numpy(); predictions[name]=p
        row=next(r for r in old_rows if r['name']==name)
        assert abs(metrics(y.iloc[va],p)['auc']-row['auc'])<1e-9
        rows.append({**row,'features':'original','reused':True})
assert 'xgb_gpu_depth7' in specs and 'cat_gpu_depth6' in specs, 'Attach prior benchmark outputs'
base_xgb=specs['xgb_gpu_depth7']['params']
base_lgb=dict(objective='binary',metric='auc',learning_rate=.025,num_leaves=47,max_depth=-1,
    min_child_samples=70,colsample_bytree=.5,subsample=.85,subsample_freq=1,reg_alpha=.4,
    reg_lambda=5.,verbosity=-1,importance_type='gain',device_type='gpu',gpu_use_dp=False,max_bin=127)
candidates=[
    dict(name='xgb_aug_depth7',kind='xgb',features='augmented',params=base_xgb),
    dict(name='xgb_peer_depth7',kind='xgb',features='peer',params=base_xgb),
    dict(name='xgb_peer_depth4',kind='xgb',features='peer',params={**base_xgb,'max_depth':4,'min_child_weight':30,'colsample_bytree':.5,'learning_rate':.035,'max_bin':256}),
    dict(name='lgb_peer_goss',kind='lgb',features='peer',params={**base_lgb,'boosting_type':'goss','num_leaves':54,'max_depth':10,'subsample':1.,'subsample_freq':0,'reg_lambda':.5}),
    dict(name='lgb_peer_gbdt',kind='lgb',features='peer',params=base_lgb),
]
(OUT/'experiment_plan.json').write_text(json.dumps(dict(candidates=candidates,baseline_auc=.7916386766841693,
    selection='One candidate per family; at most two models selected by coarse benchmark blend; 5-fold development CV; freeze weights; evaluate holdout once.',
    sources=['https://github.com/js-aguiar/home-credit-default-competition','https://github.com/NoxMoon/home-credit-default-risk']),indent=2))
for spec in candidates:
    print('BENCHMARK_START',spec['name'],flush=True)
    ps,info=fit(spec,tr,va,tag='benchmark_'+spec['name'])
    row={**info,**metrics(y.iloc[va],ps[0]),'reused':False}
    rows.append(row);specs[spec['name']]=spec;predictions[spec['name']]=ps[0]
    pd.DataFrame(rows).sort_values('auc',ascending=False).to_csv(OUT/'benchmark.csv',index=False)
    pd.DataFrame({'SK_ID_CURR':labels.iloc[va].SK_ID_CURR.to_numpy(),'TARGET':y.iloc[va].to_numpy(),**predictions}).to_csv(OUT/'benchmark_predictions.csv',index=False)
    print('BENCHMARK_RESULT',json.dumps(row),flush=True)

ranking=sorted(rows,key=lambda r:r['auc'],reverse=True)
family={}
for row in ranking:family.setdefault(row['kind'],row['name'])
bench_comparisons=compare({n:predictions[n] for n in family.values()},y.iloc[va])
chosen_bench=bench_comparisons[0]
selected=[specs[n] for n in chosen_bench['weights']]
(OUT/'benchmark_selection.json').write_text(json.dumps(dict(ranking=ranking,comparisons=bench_comparisons,selected_specs=selected),indent=2))
print('BENCHMARK_COMPLETE',json.dumps(dict(ranking=ranking,selected=chosen_bench)),flush=True)

folds=list(StratifiedKFold(CV_FOLDS,shuffle=True,random_state=SEED).split(dev,y.iloc[dev]))
fold_ids=np.full(len(dev),-1,dtype=int)
for f,(_,valid) in enumerate(folds):fold_ids[valid]=f
def cross_validate(spec,device):
    name=spec['name']; oof=np.full(len(dev),np.nan); hp=np.zeros(len(hold)); tp=np.zeros(len(XT)); fold_rows=[]; its=[]
    for f,(train,valid) in enumerate(folds):
        print('CV_START',name,f,flush=True)
        ps,info=fit(spec,dev[train],dev[valid],extras=[X.iloc[hold],XT],seed=SEED+f,tag=f'cv_{name}_{f}',device=device)
        oof[valid]=ps[0];hp+=ps[1]/CV_FOLDS;tp+=ps[2]/CV_FOLDS;its.append(info['iterations'])
        row={**info,'fold':f,**metrics(y.iloc[dev[valid]],ps[0])};fold_rows.append(row)
        pd.DataFrame(fold_rows).to_csv(OUT/f'cv_metrics_{name}.csv',index=False)
        np.savez_compressed(OUT/f'cv_predictions_{name}_{f}.npz',validation_ids=labels.iloc[dev[valid]].SK_ID_CURR.to_numpy(),val=ps[0],holdout=ps[1],test=ps[2])
        print('CV_RESULT',json.dumps(row),flush=True)
    assert np.isfinite(oof).all()
    np.savez_compressed(OUT/f'cv_complete_{name}.npz',oof=oof,holdout=hp,test=tp)
    return dict(oof=oof,holdout=hp,test=tp,iterations=its,rows=fold_rows)

with concurrent.futures.ThreadPoolExecutor(max_workers=min(2,len(selected))) as pool:
    futures={s['name']:pool.submit(cross_validate,s,i) for i,s in enumerate(selected)}
    cv={name:f.result() for name,f in futures.items()}
comparisons=compare({name:result['oof'] for name,result in cv.items()},y.iloc[dev])
chosen=comparisons[0];weights=chosen['weights']
(OUT/'frozen_selection.json').write_text(json.dumps(dict(chosen=chosen,comparisons=comparisons,note='Frozen before any holdout target evaluation; OOF scores are conditional on benchmark selection.'),indent=2))
oof=sum(w*cv[n]['oof'] for n,w in weights.items())
hp=sum(w*cv[n]['holdout'] for n,w in weights.items())
tp=sum(w*cv[n]['test'] for n,w in weights.items())
hold_metrics=metrics(y.iloc[hold],hp)
rng=np.random.default_rng(SEED); yy=y.iloc[hold].to_numpy();pos=np.flatnonzero(yy==1);neg=np.flatnonzero(yy==0);boot=[]
for _ in range(300):
    ix=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]
    boot.append(roc_auc_score(yy[ix],hp[ix]))
hold_metrics.update(auc_ci95=np.quantile(boot,[.025,.975]).tolist(),n=len(hold),positives=int(yy.sum()))
(OUT/'holdout_metrics.json').write_text(json.dumps(hold_metrics,indent=2))
pd.DataFrame({'SK_ID_CURR':labels.iloc[dev].SK_ID_CURR.to_numpy(),'TARGET':y.iloc[dev].to_numpy(),'fold':fold_ids,'prediction':oof,**{n:r['oof'] for n,r in cv.items()}}).to_csv(OUT/'oof_predictions.csv',index=False)
pd.DataFrame({'SK_ID_CURR':labels.iloc[hold].SK_ID_CURR.to_numpy(),'TARGET':yy,'prediction':hp,**{n:r['holdout'] for n,r in cv.items()}}).to_csv(OUT/'holdout_predictions.csv',index=False)
print('FROZEN_EVALUATION',json.dumps(dict(oof=chosen,holdout=hold_metrics)),flush=True)

def submission(pred,path):
    sample=pd.read_csv(CACHE/'sample_submission.csv')
    assert sample.SK_ID_CURR.is_unique and set(sample.SK_ID_CURR)==set(test_ids.SK_ID_CURR)
    sample['TARGET']=sample.SK_ID_CURR.map(pd.Series(pred,index=test_ids.SK_ID_CURR))
    assert len(sample)==48744 and sample.TARGET.notna().all() and sample.TARGET.between(0,1).all()
    assert list(sample)==['SK_ID_CURR','TARGET']
    sample.to_csv(path,index=False)
submission(tp,OUT/'submission_cv.csv')

def refit(spec,device):
    name=spec['name'];trees=max(1,int(np.median(cv[name]['iterations'])));pred=np.zeros(len(XT));refits=[]
    for seed in [42,142]:
        print('REFIT_START',name,seed,trees,flush=True)
        ps,info=fit(spec,np.arange(len(y)),extras=[XT],seed=seed,trees=trees,tag=f'final_{name}_{seed}',device=device)
        pred+=ps[0]/2;refits.append(info)
        np.save(OUT/f'final_prediction_{name}_{seed}.npy',ps[0])
        print('REFIT_RESULT',json.dumps(info),flush=True)
    return pred,refits
with concurrent.futures.ThreadPoolExecutor(max_workers=len(weights)) as pool:
    futures={s['name']:pool.submit(refit,s,i) for i,s in enumerate(s for s in selected if s['name'] in weights)}
    final={name:f.result() for name,f in futures.items()}
pred=sum(w*final[n][0] for n,w in weights.items())
submission(pred,OUT/'submission.csv')
summary=dict(environment=environment,features=manifest['features'],added_features=manifest['features']-manifest['original_feature_count'],
    benchmark_ranking=ranking,selected_specs=selected,weights=weights,oof=metrics(y.iloc[dev],oof),holdout=hold_metrics,
    final_fits={n:r[1] for n,r in final.items()},seconds=time.time()-START,submission_rows=len(pred),
    limitations=['Single development split for hyperparameter selection; OOF is conditional on selection.',
                 'Holdout follows pre-existing split; broader historical use of this dataset cannot be audited.',
                 'This notebook does not submit to the leaderboard.'])
(OUT/'run_summary.json').write_text(json.dumps(summary,indent=2))
print('RESEARCH_COMPLETE',json.dumps(summary),flush=True)
