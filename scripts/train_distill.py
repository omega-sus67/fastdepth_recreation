#!/usr/bin/env python3
"""One-epoch supervised and teacher-guided recovery on scene-grouped data."""
import argparse,hashlib,json,os,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,torch
from fastdepth_project.data import load_h5,preprocess
from fastdepth_project.models import load_converted,sha256

def masked_image_mae(prediction,target):
 if prediction.shape!=target.shape:raise ValueError('Prediction and target shape mismatch')
 valid=torch.isfinite(target)&(target>0)
 if not bool(valid.flatten(1).any(dim=1).all()):raise ValueError('Image with no valid target')
 safe_target=torch.where(valid,target,prediction.detach())
 return (prediction-safe_target).abs().flatten(1).sum(1)/valid.flatten(1).sum(1)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
 p.add_argument('--teacher',type=Path,required=True);p.add_argument('--distill-weight',type=float,default=0.5)
 p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
 p.add_argument('--smoke-samples',type=int,help='Run a limited training smoke test; not a full comparison')
 p.add_argument('--deterministic',action='store_true',help='Request deterministic PyTorch operations; CUDA requires CUBLAS_WORKSPACE_CONFIG set before process launch')
 a=p.parse_args()
 if a.output.exists():p.error('output exists')
 if not 0 <= a.distill_weight <= 1:p.error('distill weight must be between 0 and 1')
 cfg=json.loads(a.config.read_text());setting=cfg['training'];manifest=ROOT/cfg['manifest'];overlap=ROOT/cfg['overlap_report']
 m=json.loads(manifest.read_text());o=json.loads(overlap.read_text())
 if o['manifest_sha256']!=sha256(manifest) or not o['train_tune_scene_disjoint'] or o['exact_rgb_overlap_with_evaluation'] or o['exact_rgb_overlap_train_tune_count']:raise ValueError('Overlap audit failed')
 if a.seed not in setting['seeds']:raise ValueError('Seed not in protocol')
 if setting['epochs']!=1 or not setting['freeze_batch_norm_statistics']:raise ValueError('Unsupported training protocol')
 if a.deterministic:
  if a.device.startswith('cuda') and os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8',':16:8'):
   raise ValueError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before launching this process')
  torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 rows=m['groups']['train']['samples']
 if a.smoke_samples is not None:
  if a.smoke_samples<1:raise ValueError('smoke sample count must be positive')
  rows=rows[:a.smoke_samples]
 if not rows:raise ValueError('Empty training data')
 for row in rows:
  if sha256(ROOT/row['path'])!=row['sha256']:raise ValueError(f"Sample hash mismatch: {row['path']}")
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
 if a.device.startswith('cuda'):torch.cuda.manual_seed_all(a.seed)
 generator=torch.Generator().manual_seed(a.seed);order=torch.randperm(len(rows),generator=generator).tolist()
 order_hash=hashlib.sha256(('\n'.join(rows[i]['path'] for i in order)+'\n').encode()).hexdigest()
 model,record=load_converted(a.checkpoint,a.device);model.train()
 teacher,_=load_converted(a.teacher,a.device);teacher.eval()
 for layer in model.modules():
  if isinstance(layer,torch.nn.BatchNorm2d):layer.eval()
 opt=torch.optim.Adam(model.parameters(),lr=setting['learning_rate']);batch_size=setting['batch_size'];loss_total=0.;steps=0
 a.output.mkdir(parents=True);start=time.monotonic()
 for offset in range(0,len(order),batch_size):
  indices=order[offset:offset+batch_size];xs=[];ys=[]
  for index in indices:
   rgb,depth=load_h5(ROOT/rows[index]['path']);x,y=preprocess(rgb,depth);xs.append(x);ys.append(y)
  x=torch.stack(xs).to(a.device);y=torch.stack(ys).to(a.device)
  opt.zero_grad(set_to_none=True);pred=model(x)
  supervised=masked_image_mae(pred,y)
  with torch.no_grad(): teacher_output=teacher(x)
  if not bool(torch.isfinite(teacher_output).all()):raise ValueError('Nonfinite teacher output')
  valid=torch.isfinite(y)&(y>0)
  distill=torch.where(valid,(pred-teacher_output).abs(),torch.zeros_like(pred)).flatten(1).sum(1)/valid.flatten(1).sum(1)
  losses=(1-a.distill_weight)*supervised+a.distill_weight*distill
  loss=losses.mean()
  if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite loss')
  loss.backward();opt.step();loss_total+=float(losses.detach().sum());steps+=1
  if steps%100==0:print(f'{min(offset+batch_size,len(rows))}/{len(rows)} training images',flush=True)
 ckpt={'schema_version':1,'variant':record['variant'],'model_config':record['model_config'],
   'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'provenance':record['provenance'],
   'optimizer':opt.state_dict(),'epoch':1,'rng_cpu':torch.get_rng_state(),'rng_shuffle':generator.get_state(),
   'rng_cuda':torch.cuda.get_rng_state() if a.device.startswith('cuda') else None,
   'experiment':{'manifest_sha256':sha256(manifest),'initial_sha256':sha256(a.checkpoint),'seed':a.seed,
     'learning_rate':setting['learning_rate'],'batch_size':batch_size,'freeze_bn':True,'order_sha256':order_hash,'deterministic':a.deterministic,'teacher_sha256':sha256(a.teacher),'distill_weight':a.distill_weight}}
 torch.save(ckpt,a.output/'checkpoint.pth')
 report={'scope':'partial smoke, not comparison' if a.smoke_samples is not None else 'one-epoch training only; tuning not evaluated here',
   'config_sha256':sha256(a.config),'manifest_sha256':sha256(manifest),'initial_checkpoint_sha256':sha256(a.checkpoint),
   'result_checkpoint_sha256':sha256(a.output/'checkpoint.pth'),'seed':a.seed,'device':a.device,
   'gpu':torch.cuda.get_device_name(0) if a.device.startswith('cuda') else None,'torch':str(torch.__version__),
   'training_samples':len(rows),'batch_size':batch_size,'optimizer_steps':steps,'learning_rate':setting['learning_rate'],
   'optimizer':'Adam','freeze_batch_norm_statistics':True,'order_sha256':order_hash,'deterministic_requested':a.deterministic,
   'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG') if a.deterministic else None,
   'teacher_sha256':sha256(a.teacher),'distill_weight':a.distill_weight,'train_mean_combined_loss':loss_total/len(rows),'elapsed_seconds':time.monotonic()-start,
   'limitations':'Teacher is fixed in evaluation mode; distillation uses valid target pixels only. BN eval mode changes statistics used and buffer updates; pretraining and evaluation scene identities are unknown. This script saves state but does not implement resume.'}
 (a.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
