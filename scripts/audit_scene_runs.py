#!/usr/bin/env python3
"""Check the expected paired scene-experiment recovery runs and their hashes."""
import argparse,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
from hashlib import sha256

def digest(path):return sha256(path.read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--candidates',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error('output exists')
 cfg=json.loads(a.config.read_text());candidates=json.loads(a.candidates.read_text());errors=[];runs=[]
 if candidates['config_sha256']!=digest(a.config):raise ValueError('Candidate protocol mismatch')
 manifest=ROOT/cfg['manifest'];manifest_hash=digest(manifest);sample_count=len(json.loads(manifest.read_text())['groups']['train']['samples'])
 for seed in cfg['training']['seeds']:
  hashes=set()
  for method in ['original','weight','average_ablation','scene_aware','random']:
   folder=ROOT/f'results/training/2026-11-14-scene-aware-{method}-{seed}';summary=folder/'summary.json';checkpoint=folder/'checkpoint.pth'
   if not summary.exists() or not checkpoint.exists():errors.append(f'missing {method} {seed}');continue
   r=json.loads(summary.read_text());source=ROOT/cfg['source_checkpoint'] if method=='original' else ROOT/candidates['models'][f'random_{seed}' if method=='random' else method]['path']
   checks={'seed':r['seed']==seed,'config':r['config_sha256']==digest(a.config),'manifest':r['manifest_sha256']==manifest_hash,
    'source':r['initial_checkpoint_sha256']==digest(source),'output':r['result_checkpoint_sha256']==digest(checkpoint),
    'sample_count':r['training_samples']==sample_count,'batch':r['batch_size']==cfg['training']['batch_size'],
    'freeze_bn':r['freeze_batch_norm_statistics'] is True,'finite_loss':math.isfinite(r['train_mean_image_mae'])}
   if not all(checks.values()):errors.append(f'{method} {seed}: '+','.join(k for k,v in checks.items() if not v))
   hashes.add(r['order_sha256']);runs.append({'method':method,'seed':seed,'summary':str(summary.relative_to(ROOT)),
    'summary_sha256':digest(summary),'checkpoint_sha256':digest(checkpoint),'checks':checks})
  if len(hashes)!=1:errors.append(f'seed {seed} order mismatch')
 out={'scope':'Run pairing and saved hash checks; no claim of full determinism, independent data or correctness of training gradients',
  'config_sha256':digest(a.config),'candidate_manifest_sha256':digest(a.candidates),'manifest_sha256':manifest_hash,
  'run_count':len(runs),'expected':15,'same_order_per_seed':not any('order mismatch' in e for e in errors),'runs':runs,'errors':errors}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'runs':len(runs),'errors':errors},indent=2))
 if errors or len(runs)!=15:raise SystemExit(1)

if __name__=='__main__':main()
