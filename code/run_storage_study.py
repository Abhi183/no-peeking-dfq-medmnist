"""Fixed test-only study. Compression never receives data or test predictions."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score
from torch.utils.data import DataLoader,TensorDataset
from model import build_model
from data import _to_tensors
from evaluate_quant import _quantizable_names
from corrected_eval import _apply_fixed
from quantizers import lloyd_max_codebook,quant_fp16,quant_int8_minmax
from saved_codec import save_model,load_model,replace_raw_parameters
from matched_dfq import prepare_dfq,bias_correct

CONFIGS=[('fp32',32),('fp16',16),('int8_minmax',8)]+[(m,b) for b in (2,3,4) for m in (('uniform_full','rotated_normal_codebook') if b!=4 else ('uniform_full','rotated_uniform','normal_codebook','rotated_normal_codebook'))]

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def metrics(y,p):
    auc=roc_auc_score(y,p[:,1]) if p.shape[1]==2 else roc_auc_score(y,p,multi_class='ovr',average='macro')
    return dict(acc=accuracy_score(y,p.argmax(1)),auc=auc,f1_macro=f1_score(y,p.argmax(1),average='macro',zero_division=0))

@torch.inference_mode()
def predict(model,loader,device):
    model=model.to(device).eval()
    logits=torch.cat([model(x.to(device)).cpu() for x,_ in loader])
    return logits.numpy(),torch.softmax(logits,1).numpy()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--checkpoint',nargs='*',type=Path)
    a=ap.parse_args(); out=a.output; out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.csv').exists():
        raise FileExistsError('Use a fresh output directory; existing results will not be overwritten or reused.')
    for d in ('packed','predictions'): (out/d).mkdir(exist_ok=True)
    torch.set_num_threads(4)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    protocol={'device':device,'torch':torch.__version__,'numpy':np.__version__,'python':platform.python_version(),
      'configs':CONFIGS,'rotation_seed':0,'batch_size':256,'bootstrap_seed':20260910,
      'data_access':'Only test_images/test_labels loaded; compression receives model state only.',
      'dfq':'BN fold + 13 safe ReLU CLE pairs + analytic BC on 2 pointwise convs; no high bias absorption, no activation quantization. Weight-only adaptation, not canonical W8A8 DFQ.',
      'scale_precision':'FP16','excluded_parameters':'FP16','buffers':'original dtype'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    paths=a.checkpoint or sorted((Path(__file__).parent/'checkpoints').glob('*.pt'))
    print('Building deterministic codebooks',flush=True)
    cb={b:lloyd_max_codebook(2**b) for b in (2,3,4)}
    rows=[]; audits=[]
    old=out/'metrics.csv'
    for path in paths:
        ck=torch.load(path,map_location='cpu',weights_only=False)
        ds,seed=ck['dataset'],ck['seed']; base=ck['state_dict']
        with np.load(a.data/f'{ds}_64.npz') as z:
            images=z['test_images']; labels=z['test_labels']
        x,y=_to_tensors(images,labels,ck['n_channels']); yn=y.numpy()
        fingerprint=hashlib.sha256(images.tobytes()+labels.tobytes()).hexdigest()
        loader=DataLoader(TensorDataset(x,y),batch_size=256,shuffle=False)
        original=build_model(ck['n_classes'],ck['n_channels'],False).eval(); original.load_state_dict(base)
        names=_quantizable_names(original); protected={names[0],names[-1]}
        def score(label,bits,state,topology='original',artifact=None):
            key=f'{ds}_{seed}_{label}_{bits}'
            if any(r['key']==key for r in rows): return
            fresh=build_model(ck['n_classes'],ck['n_channels'],False).eval()
            if topology=='folded': fresh,_,_=prepare_dfq(fresh,equalize=False)
            fresh.load_state_dict(state)
            t=time.time(); logits,probs=predict(fresh,loader,device)
            if not np.isfinite(probs).all(): raise ValueError('nonfinite prediction')
            pred=out/'predictions'/f'{key}.npz'
            np.savez_compressed(pred,index=np.arange(len(y)),label=yn,logits=logits,probs=probs,
                dataset=ds,model_seed=seed,test_sha256=fingerprint,checkpoint_sha256=sha(path))
            rows.append(dict(key=key,dataset=ds,model_seed=seed,method=label,bits=bits,
                packed_bytes=artifact.stat().st_size if artifact else 0,
                packed_sha256=sha(artifact) if artifact else '',prediction_sha256=sha(pred),
                checkpoint_sha256=sha(path),test_sha256=fingerprint,**metrics(yn,probs)))
            pd.DataFrame(rows).to_csv(old,index=False)
            print(key,metrics(yn,probs),f'{time.time()-t:.1f}s',flush=True)
        for method,bits in CONFIGS:
            # Reproduce the historical float-scale reconstruction in this runtime.
            if method=='fp32': state=base
            elif method in ('fp16','int8_minmax'):
                q=quant_fp16 if bits==16 else quant_int8_minmax
                state={k:v.clone() for k,v in base.items()}
                for name in names: state[name]=torch.from_numpy(q(base[name].numpy()).weight)
            else:
                state,_,_=_apply_fixed(base,names,method,bits,protected=protected,codebook=cb[bits],rotation_seed=0)
            score('archive_'+method,bits,state)
            file=out/'packed'/f'{ds}_{seed}_{method}_{bits}.qbin'
            save_model(file,original,names,method,bits,cb if 'codebook' in method else {},protected if bits<8 else set())
            decoded,meta=load_model(file)
            score('saved_'+method,bits,decoded,artifact=file)
        for equalize in (False,True):
            ref,means,audit=prepare_dfq(original,equalize=equalize)
            audit.update(dataset=ds,model_seed=seed,equalize=equalize); audits.append(audit)
            score('cle_fp32' if equalize else 'folded_fp32',32,ref.state_dict(),'folded')
            for bits in (2,3,4,8):
                label='dfq_cle_bc' if equalize else 'folded_uniform'
                file=out/'packed'/f'{ds}_{seed}_{label}_{bits}.qbin'
                save_model(file,ref,names,'uniform_full' if bits<8 else 'int8_minmax',bits,{},protected if bits<8 else set())
                decoded,_=load_model(file)
                if equalize:
                    corrected=bias_correct(ref,decoded,means)
                    replace_raw_parameters(file,{n+'.bias':corrected[n+'.bias'] for n in means})
                    decoded,_=load_model(file)
                score(label,bits,decoded,'folded',file)
        (out/'dfq_audit.json').write_text(json.dumps(audits,indent=2))
    print('COMPLETE',len(rows),flush=True)

if __name__=='__main__': main()
