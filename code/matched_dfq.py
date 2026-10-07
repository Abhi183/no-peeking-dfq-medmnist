"""Explicit weight-only DFQ adaptation: BN folding, safe ReLU CLE, analytic BC.

Not full W8A8 DFQ. No high-bias absorption, no activation replacement or
quantization. Equalize only proven Conv-ReLU-Conv paths, including SE fc1/fc2.
Analytic BC restricted to two pointwise convolutions immediately after BN/ReLU:
this avoids padding and unsupported hardswish/SE input distribution assumptions.
"""
import copy
import numpy as np
import torch
from torch import nn
from scipy.special import ndtr

@torch.no_grad()
def prepare_dfq(model,equalize=True):
    m=copy.deepcopy(model).cpu().eval()
    stats={}; folded=[]
    for name,block in list(m.named_modules()):
        if isinstance(block,nn.Sequential) and len(block)>=2 and isinstance(block[0],nn.Conv2d) and isinstance(block[1],nn.BatchNorm2d):
            bn=block[1]
            stats[name+'.0']=[bn.bias.double().numpy().copy(),bn.weight.double().abs().numpy().copy()]
            block[0]=torch.nn.utils.fusion.fuse_conv_bn_eval(block[0],bn)
            block[1]=nn.Identity(); folded.append(name+'.0')
    pairs=[]
    for i in (2,3):
        for j in (0,1): pairs.append((f'features.{i}.block.{j}.0',f'features.{i}.block.{j+1}.0'))
    for name,mod in m.named_modules():
        if hasattr(mod,'fc1') and hasattr(mod,'fc2') and isinstance(getattr(mod,'activation',None),nn.ReLU):
            pairs.append((name+'.fc1',name+'.fc2'))
    modules=dict(m.named_modules())
    iterations=0
    if equalize:
        for iterations in range(1,101):
            largest=0.
            for left,right in pairs:
                a,b=modules[left],modules[right]
                r1=a.weight.abs().flatten(1).amax(1)
                if b.groups==b.in_channels==b.out_channels:
                    r2=b.weight.abs().flatten(1).amax(1)
                else:
                    assert b.groups==1
                    r2=b.weight.abs().amax(dim=(0,2,3))
                s=torch.ones_like(r1)
                active=(r1>1e-12)&(r2>1e-12)
                s[active]=torch.sqrt(r1[active]/r2[active]).clamp(1e-3,1e3)
                largest=max(largest,float(torch.abs(torch.log(s)).max()))
                a.weight.div_(s[:,None,None,None]); a.bias.div_(s)
                if b.groups==b.in_channels==b.out_channels: b.weight.mul_(s[:,None,None,None])
                else: b.weight.mul_(s[None,:,None,None])
                if left in stats:
                    stats[left][0]/=s.double().numpy(); stats[left][1]/=s.double().numpy()
            if largest<1e-5: break
    means={}
    for i in (2,3):
        beta,sigma=stats[f'features.{i}.block.1.0']
        z=np.divide(beta,sigma,out=np.zeros_like(beta),where=sigma>0)
        expectation=sigma*np.exp(-z*z/2)/np.sqrt(2*np.pi)+beta*ndtr(z)
        expectation=np.where(sigma>0,expectation,np.maximum(beta,0))
        means[f'features.{i}.block.2.0']=expectation
    return m,means,{'folded':folded,'pairs':pairs if equalize else [],'iterations':iterations,
                    'bias_correction_layers':list(means) if equalize else []}

@torch.no_grad()
def bias_correct(reference,decoded,means):
    state={k:v.clone() for k,v in decoded.items()}
    for name,mean in means.items():
        delta=(state[name+'.weight']-reference.state_dict()[name+'.weight']).double().numpy()
        correction=(delta[:,:,0,0]@mean).astype(np.float32)
        state[name+'.bias']-=torch.from_numpy(correction)
    return state
