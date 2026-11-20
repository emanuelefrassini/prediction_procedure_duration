import numpy as np
import pytest
import torch
from modern_etc.data import folds, synthetic_cases, collate, baseline_fit
from modern_etc.models import ARCHITECTURES, ETCModel, masked_mae
from modern_etc.benchmark import metrics, run

torch.set_num_threads(1)

@pytest.mark.parametrize('architecture',ARCHITECTURES)
def test_causality_and_gradients(architecture):
    torch.manual_seed(1); m=ETCModel(architecture,8).eval()
    x=torch.tensor([[0,1,2,3,4,5]])
    z=torch.tensor([[0,1,2,11,13,-1]])
    a=m(x); b=m(z); c=m(x[:,:3]); assert a.shape==(1,6)
    torch.testing.assert_close(a[:,:3],b[:,:3],atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(a[:,:3],c,atol=2e-6,rtol=2e-6)
    assert (a>=0).all()
    a.sum().backward(); assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_splits():
    seen=[]
    for tr,va,te in folds(31,3,5):
        assert not(set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))
        assert len(set(tr)|set(va)|set(te))==31
        seen.extend(te)
    assert sorted(seen)==list(range(31))


def test_padding_and_metrics():
    y=torch.tensor([[600.,0.,-1.],[300.,-1.,-1.]])
    p=torch.tensor([[1.,0.,999.],[1.,999.,999.]])
    assert float(masked_mae(p,y))==pytest.approx(1/6)
    s=metrics(y,p*600); assert s['pooled_mae_seconds']==100
    assert s['macro_mae_seconds']==150
    assert s['paper_style_smape_percent']==pytest.approx(100/9)


def test_data_and_baseline():
    cases=synthetic_cases(8,7,10); other=synthetic_cases(8,7,10)
    for (p,y),(q,z) in zip(cases,other):
        assert np.array_equal(p,q) and np.array_equal(y,z)
        assert y[-1]==0 and np.all(np.diff(y)==-10)
    x,y=collate(cases); assert torch.equal(x==-1,y==-1)
    assert np.isfinite(baseline_fit(cases)).all()


def test_end_to_end():
    kw=dict(n=12,epochs=1,count=2,width=8,seed=2)
    a=run(**kw); b=run(**kw)
    assert a==b
    assert set(a['mean_fold_metrics'])==set(ARCHITECTURES)|{'phase_mean_baseline','fixed_two_model_ensemble'}
    for m in a['mean_fold_metrics'].values(): assert np.isfinite(list(m.values())).all()


def test_invalid_metric_inputs():
    with pytest.raises(ValueError): metrics(np.array([[float('nan'), 0.]]), np.zeros((1,2)))
    with pytest.raises(ValueError): metrics(np.array([[-2., 0.]]), np.zeros((1,2)))
    with pytest.raises(ValueError): masked_mae(torch.zeros(2), torch.zeros(3))
