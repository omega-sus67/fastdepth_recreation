#!/usr/bin/env python3
"""Evaluate a locally built desktop TensorRT engine on the same validation files."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,torch,tensorrt as trt
from fastdepth_project.data import load_h5,preprocess
from fastdepth_project.metrics import DepthMetrics
from fastdepth_project.models import sha256
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--engine',type=Path,required=True,help='Path to an engine built for this machine')
p.add_argument('--data',type=Path,default=ROOT/'data/nyu_depth_v2/extracted/val/official')
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():p.error('output exists')
paths=sorted(a.data.glob('*.h5'))
if not paths:p.error('no samples')
a.output.mkdir(parents=True)
logger=trt.Logger(trt.Logger.ERROR);runtime=trt.Runtime(logger)
engine=runtime.deserialize_cuda_engine(a.engine.read_bytes())
if engine is None:raise RuntimeError('cannot load locally built TensorRT engine')
context=engine.create_execution_context();x=torch.empty((1,3,224,224),device='cuda');y=torch.empty((1,1,224,224),device='cuda')
if not context.set_tensor_address('rgb',x.data_ptr()) or not context.set_tensor_address('depth',y.data_ptr()):raise RuntimeError('binding failed')
stream=torch.cuda.current_stream().cuda_stream;total=DepthMetrics();rows=[]
for i,path in enumerate(paths,1):
 rgb,depth=load_h5(path);sample,target=preprocess(rgb,depth);x.copy_(sample.unsqueeze(0))
 if not context.execute_async_v3(stream):raise RuntimeError('inference failed')
 torch.cuda.synchronize();prediction=y.cpu().numpy()[0,0];truth=target.numpy()[0]
 metric=DepthMetrics();metric.update(prediction,truth);total.update(prediction,truth);rows.append({'sample':path.name,**metric.result()})
 if i%50==0 or i==len(paths):print(f'{i}/{len(paths)}',flush=True)
with (a.output/'per_image.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
manifest='\n'.join(path.name for path in paths)+'\n';(a.output/'manifest.txt').write_text(manifest)
report={'scope':'Locally supplied TensorRT engine; local validation set; precision, units and split identity require verification',
 'engine_sha256':sha256(a.engine),'tensorrt':trt.__version__,'gpu':torch.cuda.get_device_name(0),
 'samples':len(paths),'manifest_sha256':hashlib.sha256(manifest.encode()).hexdigest(),
 'preprocessing':'same paired preprocessing as PyTorch baseline','depth_units':'native units unverified',
 'pooled':total.result(),'mean_image_rmse':float(np.mean([r['rmse'] for r in rows])),
 'mean_image_delta1':float(np.mean([r['delta1'] for r in rows])),
 'precision_note':'FP16 tactics enabled; other operations may use FP32'}
(a.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))
