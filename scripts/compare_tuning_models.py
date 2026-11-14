#!/usr/bin/env python3
"""Compare existing checkpoints on a recorded scene-grouped tuning manifest."""
import argparse,csv,hashlib,json,platform,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from fastdepth_project.data import load_h5,preprocess
from fastdepth_project.metrics import DepthMetrics
from fastdepth_project.models import load_converted,sha256

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');a=p.parse_args()
 if a.output.exists():p.error('output exists; choose a new path')
 cfg=json.loads(a.config.read_text());manifest_path=ROOT/cfg['manifest'];overlap_path=ROOT/cfg['overlap_report']
 m=json.loads(manifest_path.read_text());overlap=json.loads(overlap_path.read_text())
 if cfg['group']!='tune':raise ValueError('This script evaluates tuning data only')
 if overlap['manifest_sha256']!=sha256(manifest_path) or not overlap['train_tune_scene_disjoint'] or overlap['exact_rgb_overlap_with_evaluation'] or overlap['exact_rgb_overlap_train_tune_count']:
  raise ValueError('Missing or failed overlap check')
 samples=m['groups']['tune']['samples']
 if not samples:raise ValueError('Empty tuning group')
 models={}
 for item in cfg['models']:
  name=item['name'];path=ROOT/item['checkpoint']
  if name in models:raise ValueError('Duplicate model name')
  model,record=load_converted(path,a.device);model.eval()
  models[name]={'model':model,'path':item['checkpoint'],'description':item['description'],
   'checkpoint_sha256':sha256(path),'parameters':sum(v.numel() for v in model.parameters()),
   'variant':record['variant'],'total':DepthMetrics(),'rmse':[],'scenes':{}}
 if not models:raise ValueError('No models selected')
 a.output.mkdir(parents=True);start=time.monotonic();rows=[]
 with torch.inference_mode():
  for n,row in enumerate(samples,1):
   path=ROOT/row['path']
   if sha256(path)!=row['sha256']:raise ValueError(f'Sample hash mismatch: {path}')
   scene=Path(row['source_member']).parts[-2]
   rgb,depth=load_h5(path);x,y=preprocess(rgb,depth);x=x.unsqueeze(0).to(a.device);target=y.numpy()[0]
   for name,entry in models.items():
    prediction=entry['model'](x).cpu().numpy()[0,0]
    one=DepthMetrics();one.update(prediction,target);result=one.result()
    entry['total'].update(prediction,target);entry['rmse'].append(result['rmse'])
    scene_data=entry['scenes'].setdefault(scene,{'metrics':DepthMetrics(),'rmse':[]})
    scene_data['metrics'].update(prediction,target);scene_data['rmse'].append(result['rmse'])
    rows.append({'model':name,'scene':scene,'sample':row['path'],**result})
   if n%50==0 or n==len(samples):print(f'{n}/{len(samples)} tuning samples',flush=True)
 with (a.output/'per_image.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 result_models={}
 for name,entry in models.items():
  scenes={scene:{'samples':len(v['rmse']),'mean_image_rmse':float(np.mean(v['rmse'])),'pooled':v['metrics'].result()}
          for scene,v in sorted(entry['scenes'].items())}
  result_models[name]={'checkpoint':entry['path'],'checkpoint_sha256':entry['checkpoint_sha256'],
   'description':entry['description'],'variant':entry['variant'],'parameters':entry['parameters'],
   'pooled':entry['total'].result(),'mean_image_rmse':float(np.mean(entry['rmse'])),
   'mean_scene_rmse':float(np.mean([v['mean_image_rmse'] for v in scenes.values()])),
   'worst_scene':max(scenes,key=lambda s:scenes[s]['mean_image_rmse']),'scenes':scenes}
 report={'scope':'Exploratory comparison on tuning split only; not final evaluation or Orin',
  'config_sha256':sha256(a.config),'manifest_sha256':sha256(manifest_path),'overlap_sha256':sha256(overlap_path),
  'samples':len(samples),'named_scenes':len(m['groups']['tune']['scenes']),'device':a.device,
  'gpu':torch.cuda.get_device_name(0) if a.device.startswith('cuda') else None,'torch':str(torch.__version__),
  'python':platform.python_version(),'precision':'PyTorch FP32; no TensorRT engine',
  'preprocessing':'same project load_h5/preprocess as prior training/evaluation','depth_units':'native mirror units; conversion chain not independently verified',
  'metric_notes':'positive finite target mask; pooled pixel metrics; equal-weight mean-image and mean-scene RMSE; nonpositive predictions handled by DepthMetrics',
  'elapsed_seconds':time.monotonic()-start,'limitations':cfg['limitations'],'models':result_models}
 (a.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({name:{'pooled_rmse':v['pooled']['rmse'],'mean_image_rmse':v['mean_image_rmse'],
                       'delta1':v['pooled']['delta1'],'mean_scene_rmse':v['mean_scene_rmse'],
                       'worst_scene':v['worst_scene']} for name,v in result_models.items()},indent=2))

if __name__=='__main__':main()
