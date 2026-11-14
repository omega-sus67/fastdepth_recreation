#!/usr/bin/env python3
"""Run a fixed-budget scene-grouped exploratory fine-tuning experiment."""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,torch
from fastdepth_project.data import load_h5,preprocess
from fastdepth_project.metrics import DepthMetrics
from fastdepth_project.models import load_converted,sha256
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
p.add_argument('--overlap-report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--epochs',type=int,default=1);p.add_argument('--seed',type=int,default=42)
p.add_argument('--learning-rate',type=float,default=1e-5);p.add_argument('--resume',type=Path)
p.add_argument('--freeze-bn',action='store_true',help='Use stored batch-normalization statistics and stop updating them; affine parameters still train')
p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');a=p.parse_args()
if a.output.exists():p.error('output exists')
if a.epochs<1 or a.learning_rate<=0:p.error('positive epochs and learning rate required')
manifest=json.loads(a.manifest.read_text());overlap=json.loads(a.overlap_report.read_text())
if overlap['manifest_sha256']!=sha256(a.manifest) or not overlap['train_tune_scene_disjoint'] or overlap['exact_rgb_overlap_with_evaluation'] or overlap['exact_rgb_overlap_train_tune_count']:
 raise ValueError('Missing or failed overlap check for the selected manifest')
groups={}
for label in ('train','tune'):
 entries=manifest['groups'][label]['samples'];groups[label]=[]
 for item in entries:
  path=ROOT/item['path']
  if sha256(path)!=item['sha256']:raise ValueError(f'Manifest hash mismatch: {path}')
  groups[label].append(path)
 if not groups[label]:raise ValueError('Empty data group')
torch.manual_seed(a.seed)
if a.device.startswith('cuda'):torch.cuda.manual_seed_all(a.seed)
generator=torch.Generator().manual_seed(a.seed)
model,record=load_converted(a.checkpoint,a.device);opt=torch.optim.Adam(model.parameters(),lr=a.learning_rate)
start=0
if a.resume:
 ckpt=torch.load(a.resume,map_location='cpu',weights_only=True)
 if ckpt['experiment']['manifest_sha256']!=sha256(a.manifest) or ckpt['experiment']['initial_sha256']!=sha256(a.checkpoint) or ckpt['experiment']['seed']!=a.seed or ckpt['experiment']['learning_rate']!=a.learning_rate or ckpt['experiment'].get('freeze_bn',False)!=a.freeze_bn:raise ValueError('Resume configuration differs')
 model.load_state_dict(ckpt['state_dict']);opt.load_state_dict(ckpt['optimizer']);start=ckpt['epoch']
 torch.set_rng_state(ckpt['rng_cpu']);generator.set_state(ckpt['rng_shuffle'])
 if a.device.startswith('cuda'):torch.cuda.set_rng_state(ckpt['rng_cuda'])
if a.epochs<=start:raise ValueError('Target epochs must exceed saved epoch')
def evaluate():
 model.eval();m=DepthMetrics();image=[]
 with torch.inference_mode():
  for path in groups['tune']:
   rgb,depth=load_h5(path);x,y=preprocess(rgb,depth);pred=model(x.unsqueeze(0).to(a.device)).cpu().numpy()[0,0]
   target=y.numpy()[0];one=DepthMetrics();one.update(pred,target);image.append(one.result()['rmse']);m.update(pred,target)
 return {'pooled':m.result(),'mean_image_rmse':float(np.mean(image))}
a.output.mkdir(parents=True)
before=evaluate();history=[];start_time=time.monotonic()
for epoch in range(start,a.epochs):
 model.train();losses=[]
 if a.freeze_bn:
  for layer in model.modules():
   if isinstance(layer,torch.nn.BatchNorm2d):layer.eval()
 for idx in torch.randperm(len(groups['train']),generator=generator).tolist():
  rgb,depth=load_h5(groups['train'][idx]);x,y=preprocess(rgb,depth);x=x.unsqueeze(0).to(a.device);y=y.unsqueeze(0).to(a.device)
  opt.zero_grad(set_to_none=True);pred=model(x);valid=torch.isfinite(y)&(y>0)
  if not bool(valid.any()):raise ValueError('no valid target pixels')
  loss=(pred[valid]-y[valid]).abs().mean()
  if not bool(torch.isfinite(loss)):raise ValueError('nonfinite training loss')
  loss.backward();opt.step();losses.append(float(loss.detach()))
 tune=evaluate();history.append({'epoch':epoch+1,'train_mae':float(np.mean(losses)),'tuning':tune})
 print(json.dumps(history[-1]),flush=True)
 ckpt={'schema_version':1,'variant':record['variant'],'model_config':record['model_config'],
       'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'provenance':record['provenance'],
       'optimizer':opt.state_dict(),'epoch':epoch+1,'rng_cpu':torch.get_rng_state(),'rng_shuffle':generator.get_state(),
       'rng_cuda':torch.cuda.get_rng_state() if a.device.startswith('cuda') else None,
       'experiment':{'manifest_sha256':sha256(a.manifest),'initial_sha256':sha256(a.checkpoint),'seed':a.seed,'learning_rate':a.learning_rate,'freeze_bn':a.freeze_bn}}
 torch.save(ckpt,a.output/'checkpoint.pth')
report={'scope':'exploratory tiny scene-grouped tuning only; evaluation scene identity unknown; not Orin or final accuracy',
 'initial_checkpoint_sha256':sha256(a.checkpoint),'result_checkpoint_sha256':sha256(a.output/'checkpoint.pth'),
 'manifest_sha256':sha256(a.manifest),'overlap_report_sha256':sha256(a.overlap_report),'scene_names':overlap['scene_names'],
 'device':a.device,'torch':str(torch.__version__),'seed':a.seed,'learning_rate':a.learning_rate,'optimizer':'Adam',
 'loss':'mean absolute error on finite positive targets','batch_size':1,'epochs':a.epochs,'start_epoch':start,'freeze_batch_norm_statistics':a.freeze_bn,
 'batch_norm_note':'eval mode uses stored running statistics during forward and prevents updates; affine weights remain trainable' if a.freeze_bn else 'training mode uses batch statistics and updates running statistics',
 'tuning_before':before,'history':history,'elapsed_seconds':time.monotonic()-start_time,
 'limitations':[f"Training samples: {len(groups['train'])}; tuning samples: {len(groups['tune'])}; scene coverage is recorded separately",'Validation scene overlap unknown',
                'Depth units not independently confirmed','Not a board-aware search']}
(a.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
