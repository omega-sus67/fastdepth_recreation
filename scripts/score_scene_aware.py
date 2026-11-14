#!/usr/bin/env python3
"""Score decoder channels by average and worst-scene masked-depth loss changes."""
import argparse,json,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,torch
from fastdepth_project.data import load_h5,preprocess
from fastdepth_project.models import load_converted,sha256

def scene_score(deltas,weight):
 if not deltas:raise ValueError('at least one scene required')
 return float(np.mean(deltas)+weight*max(deltas))

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
 p.add_argument('--smoke',action='store_true',help='Only 2 scenes and 3 channels; unsuitable for pruning')
 a=p.parse_args()
 if a.output.exists():p.error('output exists')
 cfg=json.loads(a.config.read_text());cal=cfg['score_calibration'];manifest=ROOT/cfg['manifest'];overlap=ROOT/cfg['overlap_report']
 m=json.loads(manifest.read_text());o=json.loads(overlap.read_text())
 if o['manifest_sha256']!=sha256(manifest) or not o['train_tune_scene_disjoint'] or o['exact_rgb_overlap_with_evaluation'] or o['exact_rgb_overlap_train_tune_count']:raise ValueError('Overlap audit failed')
 groups={}
 for row in m['groups']['train']['samples']:
  scene=Path(row['source_member']).parts[-2];groups.setdefault(scene,[]).append(row)
 eligible=sorted(s for s,rows in groups.items() if len(rows)>=cal['samples_per_scene'])
 random.Random(cal['scene_seed']).shuffle(eligible)
 count=2 if a.smoke else cal['scene_count']
 if len(eligible)<count:raise ValueError('Not enough eligible scenes')
 scenes=eligible[:count];data=[];records=[]
 for scene in scenes:
  rows=sorted(groups[scene],key=lambda r:r['source_member']);n=cal['samples_per_scene']
  indices=[round(i*(len(rows)-1)/(n-1)) for i in range(n)] if n>1 else [0]
  xs=[];ys=[]
  for index in indices:
   row=rows[index];path=ROOT/row['path']
   if sha256(path)!=row['sha256']:raise ValueError(f'Hash mismatch: {path}')
   rgb,depth=load_h5(path);x,y=preprocess(rgb,depth);xs.append(x);ys.append(y)
   records.append({'scene':scene,'path':row['path'],'sha256':row['sha256']})
  x=torch.stack(xs).to(a.device);y=torch.stack(ys).to(a.device);valid=torch.isfinite(y)&(y>0)
  if not bool(valid.flatten(1).any(dim=1).all()):raise ValueError('Image with no valid targets')
  data.append((x,y,valid))
 model,_=load_converted(ROOT/cfg['source_checkpoint'],a.device);model.eval()
 width=model.decode_conv2[1][0].in_channels
 if width!=200:raise ValueError('Expected 200 input channels')
 channel=[None]
 def mask(_,inputs,output):
  if channel[0] is None:return output
  result=output.clone();result[:,channel[0]]=0
  return result
 hook=model.decode_conv2[0].register_forward_hook(mask)
 def loss_by_scene():
  values=[]
  for x,y,valid in data:
   pred=model(x);errors=torch.where(valid,(pred-y).abs(),torch.zeros_like(y))
   per_image=errors.flatten(1).sum(1)/valid.flatten(1).sum(1)
   if not bool(torch.isfinite(per_image).all()):raise ValueError('Nonfinite loss')
   values.append(float(per_image.mean().cpu()))
  return values
 start=time.monotonic();scores=[]
 try:
  with torch.inference_mode():
   baseline=loss_by_scene()
   for i in range(3 if a.smoke else width):
    channel[0]=i;ablated=loss_by_scene();delta=[float(v-b) for v,b in zip(ablated,baseline)]
    scores.append({'channel':i,'mean_delta':float(np.mean(delta)),'max_scene_delta':max(delta),
                   'scene_aware_score':scene_score(delta,cal['lambda']),'scene_deltas':dict(zip(scenes,delta))})
    if (i+1)%20==0:print(f'Scored {i+1}/{width} channels',flush=True)
 finally:hook.remove()
 report={'scope':'smoke, not valid for pruning' if a.smoke else 'training-only channel scoring; no tuning or final evaluation',
  'smoke':a.smoke,'config_sha256':sha256(a.config),'manifest_sha256':sha256(manifest),'overlap_sha256':sha256(overlap),
  'checkpoint_sha256':sha256(ROOT/cfg['source_checkpoint']),'device':a.device,'torch':str(torch.__version__),
  'scene_count':len(scenes),'images':len(records),'scenes':scenes,'calibration_samples':records,'scene_baseline_mae':dict(zip(scenes,baseline)),
  'formula':cfg['score_formula'],'lambda':cal['lambda'],'channels':scores,'elapsed_seconds':time.monotonic()-start,
  'limitations':'Single-channel effects ignore group interactions; selection is a small scene sample; candidate novelty and pretraining independence unverified.'}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'scenes':len(scenes),'images':len(records),'channels':len(scores),'elapsed_seconds':report['elapsed_seconds']},indent=2))

if __name__=='__main__':main()
