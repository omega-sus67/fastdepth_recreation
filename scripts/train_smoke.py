#!/usr/bin/env python3
"""Short, correlated-scene training smoke test; not a final model experiment."""
import argparse,hashlib,json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,torch
from fastdepth_project.data import load_h5,preprocess
from fastdepth_project.models import load_converted,sha256
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--manifest',type=Path,default=ROOT/'data/nyu_depth_v2/preliminary_2026-11-13/manifest.json')
p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/converted/community-skipadd-pruned.pth')
p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=1)
p.add_argument('--seed',type=int,default=42);p.add_argument('--learning-rate',type=float,default=1e-5)
p.add_argument('--resume',type=Path);a=p.parse_args()
if a.output.exists():p.error('output exists')
if a.epochs<1:p.error('epochs must be positive')
manifest=json.loads(a.manifest.read_text());scenes=manifest['scenes'];first=next(iter(scenes.values()),[])
if len(first)<4:p.error('at least four preliminary samples are needed')
paths=[ROOT/entry['path'] for entry in first];train=paths[:-4];tune=paths[-4:]
if set(train)&set(tune):raise ValueError('overlapping filenames')
# Detect exact byte-identical evaluation files; scene/near-duplicate overlap is unknown.
audit=json.loads((ROOT/'results/data/2026-11-13-validation-audit/summary.json').read_text())
import csv
with (ROOT/'results/data/2026-11-13-validation-audit/samples.csv').open() as f:validation_hashes={r['sha256'] for r in csv.DictReader(f)}
if any(sha256(path) in validation_hashes for path in paths):raise ValueError('exact validation file overlap')
random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
model,record=load_converted(a.checkpoint,'cuda');optimizer=torch.optim.Adam(model.parameters(),lr=a.learning_rate)
start=0
if a.resume:
 saved=torch.load(a.resume,map_location='cpu',weights_only=True);model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer']);start=saved['epoch'];torch.set_rng_state(saved['rng_cpu']);torch.cuda.set_rng_state(saved['rng_cuda'])
a.output.mkdir(parents=True);history=[]
for epoch in range(start,a.epochs):
 model.train();losses=[]
 for path in train:
  rgb,depth=load_h5(path);x,y=preprocess(rgb,depth);x=x.unsqueeze(0).cuda();y=y.unsqueeze(0).cuda()
  optimizer.zero_grad(set_to_none=True);pred=model(x);valid=torch.isfinite(y)&(y>0)
  if not bool(valid.any()):raise ValueError(f'no valid pixels in {path}')
  loss=torch.abs(pred[valid]-y[valid]).mean()
  if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
  loss.backward();optimizer.step();losses.append(float(loss.detach()))
 model.eval();tune_losses=[]
 with torch.inference_mode():
  for path in tune:
   rgb,depth=load_h5(path);x,y=preprocess(rgb,depth);pred=model(x.unsqueeze(0).cuda());y=y.unsqueeze(0).cuda();valid=torch.isfinite(y)&(y>0)
   tune_losses.append(float(torch.abs(pred[valid]-y[valid]).mean()))
 item={'epoch':epoch+1,'train_mae':float(np.mean(losses)),'tune_mae':float(np.mean(tune_losses))};history.append(item);print(item,flush=True)
 saved={'schema_version':1,'variant':record['variant'],'model_config':record['model_config'],'state_dict':model.cpu().state_dict(),
        'provenance':record['provenance'],'optimizer':optimizer.state_dict(),'epoch':epoch+1,'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state()}
 torch.save(saved,a.output/'checkpoint.pth');model.cuda()
report={'scope':'preliminary correlated-scene smoke test; not final training, pruning or generalization evidence',
 'source_manifest':str(a.manifest),'train_files':[str(x.relative_to(ROOT)) for x in train],
 'tune_files':[str(x.relative_to(ROOT)) for x in tune], 'scene_separation':False,'exact_validation_hash_overlap':False,
 'validation_scene_overlap':'unverified','seed':a.seed,'learning_rate':a.learning_rate,'optimizer':'Adam','history':history,
 'source_checkpoint_sha256':sha256(a.checkpoint),'result_checkpoint_sha256':sha256(a.output/'checkpoint.pth') if history else None}
(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
