#!/usr/bin/env python3
"""Check deterministic comparison checkpoint provenance and matched data order."""
import argparse,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True)
 p.add_argument('--comparison',type=Path,required=True);p.add_argument('--candidates',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error('output exists')
 protocol=json.loads(a.protocol.read_text());comparison=json.loads(a.comparison.read_text());candidates=json.loads(a.candidates.read_text())
 manifest=ROOT/protocol['manifest'];count=len(json.loads(manifest.read_text())['groups']['train']['samples']);runs=[];errors=[]
 for seed in protocol['training']['seeds']:
  orders=set()
  for method in ['original','weight','average_ablation','scene_aware','random']:
   name=f'{method}_{seed}';matches=[i for i in comparison['models'] if i['name']==name]
   if len(matches)!=1:errors.append('missing or duplicate '+name);continue
   checkpoint=ROOT/matches[0]['checkpoint'];summary=checkpoint.parent/'summary.json'
   if not checkpoint.is_file() or not summary.is_file():errors.append('missing files '+name);continue
   r=json.loads(summary.read_text());source=ROOT/protocol['source_checkpoint'] if method=='original' else ROOT/candidates['models'][f'random_{seed}' if method=='random' else method]['path']
   checks={'seed':r['seed']==seed,'protocol':r['config_sha256']==sha(a.protocol),'manifest':r['manifest_sha256']==sha(manifest),
    'source':r['initial_checkpoint_sha256']==sha(source),'result':r['result_checkpoint_sha256']==sha(checkpoint),
    'samples':r['training_samples']==count,'batch':r['batch_size']==protocol['training']['batch_size'],
    'deterministic':r.get('deterministic_requested') is True,'cublas':r.get('cublas_workspace_config')==':4096:8',
    'bn':r['freeze_batch_norm_statistics'] is True,'loss_finite':math.isfinite(r['train_mean_image_mae'])}
   if not all(checks.values()):errors.append(name+': '+','.join(k for k,v in checks.items() if not v))
   orders.add(r['order_sha256']);runs.append({'model':name,'summary':str(summary.relative_to(ROOT)),
    'summary_sha256':sha(summary),'checkpoint_sha256':sha(checkpoint),'checks':checks})
  if len(orders)!=1:errors.append(f'data order mismatch seed {seed}')
 out={'scope':'Saved run pairing checks; no claim of cross-platform determinism or evaluation independence',
  'protocol_sha256':sha(a.protocol),'comparison_sha256':sha(a.comparison),'candidate_manifest_sha256':sha(a.candidates),
  'manifest_sha256':sha(manifest),'expected_runs':15,'runs':runs,'errors':errors}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'runs':len(runs),'errors':errors},indent=2))
 if errors or len(runs)!=15:raise SystemExit(1)

if __name__=='__main__':main()
