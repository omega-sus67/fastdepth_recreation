#!/usr/bin/env python3
"""Score first-decoder channels by individual masked-loss ablation on training images."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from fastdepth_project.data import load_h5,preprocess
from fastdepth_project.models import load_converted,sha256

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/converted/community-skipadd-pruned.pth')
p.add_argument('--manifest',type=Path,required=True)
p.add_argument('--overlap-report',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
a=p.parse_args()
if a.output.exists():p.error('output exists')
m=json.loads(a.manifest.read_text());overlap=json.loads(a.overlap_report.read_text())
if overlap['manifest_sha256']!=sha256(a.manifest) or not overlap['train_tune_scene_disjoint'] or overlap['exact_rgb_overlap_with_evaluation'] or overlap['exact_rgb_overlap_train_tune_count']:raise ValueError('Overlap check failed')
images=[]
for row in m['groups']['train']['samples']:
 path=ROOT/row['path']
 if sha256(path)!=row['sha256']:raise ValueError('Training image hash mismatch')
 rgb,depth=load_h5(path);x,y=preprocess(rgb,depth);x=x.unsqueeze(0).to(a.device);y=y.unsqueeze(0).to(a.device)
 valid=torch.isfinite(y)&(y>0)
 if not bool(valid.any()):raise ValueError('No valid depth')
 images.append((x,y,valid))
if not images:raise ValueError('No training images')
model,_=load_converted(a.checkpoint,a.device);model.eval()
count=model.decode_conv2[1][0].in_channels
if count!=200:raise ValueError('Expected original 200-channel decoder')
active=[None]
def mask(_,inputs,output):
 if active[0] is None:return output
 result=output.clone();result[:,active[0]]=0
 return result
handle=model.decode_conv2[0].register_forward_hook(mask)
def mean_loss():
 total=0.0
 for x,y,valid in images:
  pred=model(x);loss=(pred[valid]-y[valid]).abs().mean()
  if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite loss')
  total+=float(loss)
 return total/len(images)
try:
 with torch.inference_mode():
  baseline=mean_loss();scores=[]
  for i in range(count):
   active[0]=i;loss=mean_loss();scores.append({'channel':i,'ablated_mean_image_mae':loss,'loss_delta':loss-baseline})
   if (i+1)%20==0:print(f'Scored {i+1}/{count}',flush=True)
finally:handle.remove()
report={'scope':'Individual training-image ablations; not joint pruning, final evaluation, or target-device timing',
 'checkpoint_sha256':sha256(a.checkpoint),'manifest_sha256':sha256(a.manifest),'overlap_report_sha256':sha256(a.overlap_report),
 'training_images':len(images),'device':a.device,'torch':str(torch.__version__),'baseline_mean_image_mae':baseline,
 'loss':'mean of image-wise MAE on finite positive targets','placement':'zero one output channel of decode_conv2[0] immediately before decode_conv2[1] pointwise convolution',
 'ranking':'ascending loss_delta then channel index; negative means individual ablation lowered training loss','channels':scores}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
