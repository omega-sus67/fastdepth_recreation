#!/usr/bin/env python3
"""Evaluate the community checkpoint on the locally extracted validation samples."""
import argparse,csv,hashlib,json,platform,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from fastdepth_project.data import load_h5,preprocess
from fastdepth_project.metrics import DepthMetrics
from fastdepth_project.models import load_converted,sha256

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/converted/community-skipadd-pruned.pth')
p.add_argument('--data',type=Path,default=ROOT/'data/nyu_depth_v2/extracted/val/official')
p.add_argument('--output',type=Path,required=True)
p.add_argument('--limit',type=int,default=None)
p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
a=p.parse_args()
paths=sorted(a.data.glob('*.h5'))
if a.limit is not None:
 if a.limit <= 0: p.error('--limit must be positive')
 paths=paths[:a.limit]
if not paths: p.error('no HDF5 samples found')
if a.output.exists(): p.error('output directory exists; choose a new run path')
a.output.mkdir(parents=True)
model,record=load_converted(a.checkpoint,a.device)
total=DepthMetrics(); rows=[]
for i,path in enumerate(paths,1):
 rgb,depth=load_h5(path); x,y=preprocess(rgb,depth)
 with torch.inference_mode(): prediction=model(x.unsqueeze(0).to(a.device)).cpu().numpy()[0,0]
 target=y.numpy()[0]; one=DepthMetrics(); one.update(prediction,target); total.update(prediction,target)
 row={'sample':path.name,**one.result()}; rows.append(row)
 if i%50==0 or i==len(paths): print(f'{i}/{len(paths)}',flush=True)
with (a.output/'per_image.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
manifest='\n'.join(x.name for x in paths)+'\n'
(a.output/'manifest.txt').write_text(manifest)
report={'scope':'PyTorch checkpoint evaluation on the local validation collection; original split identity and depth units remain unverified',
 'variant':record['variant'],'checkpoint_source':record['provenance']['source_status'],'checkpoint_sha256':sha256(a.checkpoint),
 'device':a.device,'torch':str(torch.__version__),'python':platform.python_version(),'samples':len(paths),
 'manifest_sha256':hashlib.sha256(manifest.encode()).hexdigest(),
 'preprocessing':'Pillow nearest: 480x640 -> 250x333 -> center crop 228x304 -> 224x224; RGB /255; depth float32',
 'depth_units':'unverified, values treated as native dataset units', 'pooled':total.result(),
 'mean_image_rmse':float(np.mean([r['rmse'] for r in rows])),
 'mean_image_delta1':float(np.mean([r['delta1'] for r in rows]))}
(a.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2))
