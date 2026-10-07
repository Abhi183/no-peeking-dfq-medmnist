import numpy as np
import pytest
import torch
from saved_codec import pack_codes, unpack_codes, save_model, load_model

@pytest.mark.parametrize('bits',[2,3,4,8])
def test_packed_codes_cover_levels_and_tail(bits):
    x=np.tile(np.arange(2**bits,dtype=np.uint8),3)[:-1]
    raw=pack_codes(x,bits)
    assert len(raw)==(len(x)*bits+7)//8
    np.testing.assert_array_equal(unpack_codes(raw,bits,len(x)),x)

def test_saved_scales_and_nonquantized_parameters(tmp_path):
    model=torch.nn.Sequential(torch.nn.Linear(64,8),torch.nn.BatchNorm1d(8)).eval()
    original={k:v.clone() for k,v in model.state_dict().items()}
    p=tmp_path/'weights.qbin'
    save_model(p,model,['0.weight'],'uniform_full',4,{},set())
    state,meta=load_model(p)
    assert meta['scale_dtype']=='<f2'
    assert not torch.equal(state['0.weight'],original['0.weight'])
    assert torch.equal(state['0.bias'],original['0.bias'].half().float())
    assert torch.equal(state['1.running_var'],original['1.running_var'])
    assert state['1.num_batches_tracked'].dtype==torch.int64
    model.load_state_dict(state)

def test_zero_tensor_is_finite(tmp_path):
    model=torch.nn.Linear(64,8,bias=False)
    model.weight.data.zero_()
    p=tmp_path/'zero.qbin'
    save_model(p,model,['weight'],'uniform_full',4,{},set())
    state,_=load_model(p)
    assert torch.isfinite(state['weight']).all()
    assert torch.count_nonzero(state['weight'])==0

@pytest.mark.parametrize('method',['uniform_full','rotated_uniform','normal_codebook','rotated_normal_codebook'])
@pytest.mark.parametrize('bits',[2,3,4])
def test_decode_matches_independent_quantizer_with_rounded_scale(tmp_path,method,bits):
    from corrected_eval import _quantize_tensor
    from quantizers import lloyd_max_codebook,random_orthogonal
    torch.manual_seed(19)
    model=torch.nn.Linear(64,8,bias=False)
    w=model.weight.detach().numpy();rows=w.astype(np.float64)
    cb=lloyd_max_codebook(2**bits,n_samples=10000,iters=20)
    ref=_quantize_tensor(method,w,bits,cb,rotation_seed=0,tensor_index=0,allow_rotation=True).weight
    if 'codebook' in method:
        scale=np.linalg.norm(rows,axis=1)/np.sqrt(64)
    else:
        y=rows@random_orthogonal(64,17).T if method.startswith('rotated') else rows
        scale=np.abs(y).max(1)/(2**bits-1)
    expected=ref*(scale.astype(np.float16).astype(float)/scale)[:,None]
    path=tmp_path/'quant.qbin'
    save_model(path,model,['weight'],method,bits,{bits:cb},set())
    state,_=load_model(path)
    np.testing.assert_allclose(state['weight'].numpy(),expected,atol=2e-8,rtol=1e-6)

def test_protected_int8_and_bias_replacement(tmp_path):
    from saved_codec import replace_raw_parameters
    from quantizers import quant_int8_minmax
    torch.manual_seed(3);model=torch.nn.Linear(64,8)
    w=model.weight.detach().numpy();scale=np.max(abs(w.astype(float)),axis=1)/127
    expected=quant_int8_minmax(w).weight*(scale.astype(np.float16).astype(float)/scale)[:,None]
    path=tmp_path/'protected.qbin'
    save_model(path,model,['weight'],'uniform_full',4,{}, {'weight'})
    state,_=load_model(path)
    np.testing.assert_allclose(state['weight'],expected,atol=2e-8,rtol=1e-6)
    replace_raw_parameters(path,{'bias':torch.ones(8)*.12345})
    after,_=load_model(path)
    assert torch.equal(after['weight'],state['weight'])
    assert torch.equal(after['bias'],(torch.ones(8)*.12345).half().float())

def test_corrupt_payload_rejected(tmp_path):
    model=torch.nn.Linear(4,2);p=tmp_path/'x.qbin'
    save_model(p,model,[],'fp32',32,{},set())
    p.write_bytes(p.read_bytes()[:-1])
    with pytest.raises(ValueError,match='truncated'):load_model(p)
