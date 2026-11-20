"""Run a data-free benchmark with independent validation and fixed ensemble."""
import argparse
import copy
import json
import platform
from pathlib import Path
import numpy as np
import torch
from .data import synthetic_cases, folds, collate, baseline_fit
from .models import ARCHITECTURES, ETCModel, masked_mae

SCALE = 600.
ENSEMBLE = ('inceptiontime','lstm_fcn')


def metrics(y, pred):
    y = y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
    pred = pred.detach().cpu().numpy() if torch.is_tensor(pred) else np.asarray(pred)
    if y.shape != pred.shape or y.ndim != 2:
        raise ValueError('expected aligned case-by-time arrays')
    if not np.isfinite(y).all() or np.any((y < 0) & (y != -1)):
        raise ValueError('targets must be finite, nonnegative, or -1 padding')
    valid = y >= 0
    if not np.all(valid.any(axis=1)) or not np.isfinite(pred[valid]).all() or np.any(pred[valid]<0):
        raise ValueError('invalid target or prediction')
    ae = np.abs(y[valid]-pred[valid]); den = y[valid]+pred[valid]
    ratio=np.divide(ae,den,out=np.zeros_like(ae),where=den>0)
    return {'pooled_mae_seconds':float(ae.mean()), 'macro_mae_seconds':float(np.mean([
        np.abs(t[m]-p[m]).mean() for t,p,m in zip(y,pred,valid)])),
        'paper_style_smape_percent':float(100*ratio.mean())}


def fit(model, train, val, epochs, seed, batch_size=16):
    opt=torch.optim.Adam(model.parameters(),lr=.003)
    rng=np.random.default_rng(seed)
    best=float('inf'); state=None; best_epoch=0
    for epoch in range(epochs):
        model.train()
        order=rng.permutation(len(train))
        for start in range(0,len(order),batch_size):
            x,y=collate([train[i] for i in order[start:start+batch_size]])
            opt.zero_grad(); loss=masked_mae(model(x),y,SCALE)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
        model.eval()
        with torch.no_grad():
            x,y=collate(val); score=float(masked_mae(model(x),y,SCALE))
        if score < best:
            best=score; state=copy.deepcopy(model.state_dict()); best_epoch=epoch+1
    if state is None: raise RuntimeError('no finite validation loss')
    model.load_state_dict(state)
    return best_epoch,best*SCALE


def run(n=90, epochs=5, count=3, seed=42, width=32, interval=30, architectures=ARCHITECTURES):
    if epochs<1 or not architectures or len(set(architectures))!=len(architectures):
        raise ValueError('positive epochs and distinct architectures required')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    dataset=synthetic_cases(n,seed,interval)
    output={'scope':'synthetic only; not a reproduction or clinical result', 'framework':torch.__version__, 'python':platform.python_version(), 'numpy':np.__version__,
            'config':dict(n=n,epochs=epochs,folds=count,seed=seed,width=width,interval_seconds=interval,architectures=list(architectures)), 'folds':[]}
    for i,(tr,va,te) in enumerate(folds(n,count,seed)):
        train,val,test=([dataset[j] for j in ids] for ids in (tr,va,te))
        x,y=collate(test); baseline=baseline_fit(train)
        pred_baseline=baseline[x.clamp_min(0).numpy()]
        result={'train_ids':tr.tolist(),'validation_ids':va.tolist(),'test_ids':te.tolist(),
                'models':{'phase_mean_baseline':metrics(y,pred_baseline)}}
        predictions={}
        for j,a in enumerate(architectures):
            torch.manual_seed(seed+1000*i+ARCHITECTURES.index(a))
            model=ETCModel(a,width)
            epoch,v=fit(model,train,val,epochs,seed+i)
            model.eval()
            with torch.no_grad(): predictions[a]=model(x)*SCALE
            result['models'][a]={**metrics(y,predictions[a]),'best_epoch':epoch,'validation_mae_seconds':v}
        if all(a in predictions for a in ENSEMBLE):
            result['models']['fixed_two_model_ensemble']=metrics(y,(predictions[ENSEMBLE[0]]+predictions[ENSEMBLE[1]])/2)
        output['folds'].append(result)
    keys=output['folds'][0]['models']
    output['mean_fold_metrics']={k:{m:float(np.mean([f['models'][k][m] for f in output['folds']])) for m in
                                  ('pooled_mae_seconds','macro_mae_seconds','paper_style_smape_percent')} for k in keys}
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',type=int,default=90); p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--folds',type=int,default=3); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--width',type=int,default=32); p.add_argument('--interval',type=float,default=30)
    p.add_argument('--models',nargs='+',choices=ARCHITECTURES,default=list(ARCHITECTURES))
    p.add_argument('--output',type=Path,default=Path('modern_etc_output.json'))
    a=p.parse_args(); r=run(a.cases,a.epochs,a.folds,a.seed,a.width,a.interval,a.models)
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    for name,score in r['mean_fold_metrics'].items(): print(f"{name:25s} {score['macro_mae_seconds']:8.2f} s (macro MAE)")
    print(f'Synthetic results only: {a.output}')
