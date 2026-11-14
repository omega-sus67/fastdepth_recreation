#!/usr/bin/env python3
"""Measure synchronized model-only desktop latency; excludes preprocessing/transfers."""
import argparse,csv,json,platform,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,torch
from fastdepth_project.models import load_converted,sha256
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/converted/community-skipadd-pruned.pth')
p.add_argument('--output',type=Path,required=True)
p.add_argument('--warmup',type=int,default=50);p.add_argument('--runs',type=int,default=500)
p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
a=p.parse_args()
if a.warmup<0 or a.runs<1:p.error('invalid iteration counts')
if a.output.exists():p.error('output exists')
a.output.mkdir(parents=True)
model,record=load_converted(a.checkpoint,a.device);x=torch.zeros(1,3,224,224,device=a.device)
def sync():
 if a.device.startswith('cuda'):torch.cuda.synchronize()
with torch.inference_mode():
 for _ in range(a.warmup):model(x)
 sync(); times=[]
 for _ in range(a.runs):
  sync();start=time.perf_counter_ns();model(x);sync();times.append((time.perf_counter_ns()-start)/1e6)
with (a.output/'latencies_ms.csv').open('w',newline='') as f:
 writer=csv.writer(f);writer.writerow(['iteration','latency_ms']);writer.writerows(enumerate(times))
report={'scope':'model-only; preallocated zero tensor; excludes preprocessing, transfers, capture, rendering',
 'device':a.device,'gpu':torch.cuda.get_device_name(0) if a.device.startswith('cuda') else None,
 'runtime':'PyTorch FP32','torch':str(torch.__version__),'python':platform.python_version(),
 'checkpoint_sha256':sha256(a.checkpoint),'model_source':record['provenance']['source_status'],
 'shape':[1,3,224,224],'warmup':a.warmup,'samples':a.runs,
 'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'p99_ms':float(np.percentile(times,99)),
 'comparison_note':'Desktop exploratory timing; not Orin performance or the 33 ms end-to-end target.'}
(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
