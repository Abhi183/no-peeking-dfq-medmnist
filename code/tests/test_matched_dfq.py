import torch
from model import build_model
from matched_dfq import prepare_dfq

def test_fold_and_cle_preserve_function_and_skip_hardswish():
    torch.manual_seed(7)
    m=build_model(7,3,False).eval()
    x=torch.randn(2,3,64,64)
    with torch.no_grad(): expected=m(x)
    folded,_,_=prepare_dfq(m,equalize=False)
    cle,means,audit=prepare_dfq(m,equalize=True)
    with torch.no_grad():
        torch.testing.assert_close(folded(x),expected,rtol=2e-4,atol=1e-6)
        torch.testing.assert_close(cle(x),expected,rtol=2e-4,atol=1e-6)
    assert len(audit['pairs'])==13
    assert set(means)=={'features.2.block.2.0','features.3.block.2.0'}

def test_bias_correction_removes_expected_local_quantization_error():
    import numpy as np
    from matched_dfq import bias_correct
    torch.manual_seed(11)
    model=torch.nn.Sequential(torch.nn.Conv2d(3,2,1)).eval()
    state={k:v.clone() for k,v in model.state_dict().items()}
    state['0.weight']+=.01
    mean=np.array([.5,1.,2.])
    fixed=bias_correct(model,state,{'0':mean})
    actual=torch.nn.Sequential(torch.nn.Conv2d(3,2,1)).eval();actual.load_state_dict(fixed)
    probe=torch.tensor(mean,dtype=torch.float32).reshape(1,3,1,1)
    with torch.no_grad():torch.testing.assert_close(actual(probe),model(probe),rtol=1e-6,atol=1e-7)
