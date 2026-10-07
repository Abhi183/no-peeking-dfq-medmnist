import numpy as np
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score
from paired_bootstrap import weighted_metrics

def test_weighted_metrics_match_explicit_resampling_with_ties():
    y=np.array([0,1,2,0,1,2]); p=np.array([[.5,.3,.2],[.1,.7,.2],[.2,.2,.6],[.5,.3,.2],[.1,.4,.5],[.1,.4,.5]])
    idx=np.array([0,0,1,2,3,4,5,5,5]); w=np.bincount(idx,minlength=6)[None,:]
    actual=weighted_metrics(y,p,w)[0]
    expected=[accuracy_score(y[idx],p[idx].argmax(1)),roc_auc_score(y[idx],p[idx],multi_class='ovr'),f1_score(y[idx],p[idx].argmax(1),average='macro')]
    np.testing.assert_allclose(actual,expected,atol=1e-14)

def test_binary_metrics_and_perfect_pair():
    y=np.array([0,0,1,1]);p=np.array([[.9,.1],[.5,.5],[.5,.5],[.2,.8]])
    w=np.array([[1,2,1,2],[2,1,2,1]])
    a=weighted_metrics(y,p,w)
    for i in range(2):
        assert abs(a[i,1]-roc_auc_score(y,p[:,1],sample_weight=w[i]))<1e-14
    np.testing.assert_array_equal(a-a,np.zeros_like(a))
