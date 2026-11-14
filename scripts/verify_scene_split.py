#!/usr/bin/env python3
"""Check scene grouping and exact RGB/depth overlap with local evaluation data."""
import argparse,hashlib,json
from pathlib import Path
import h5py,numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():p.error('output exists')
m=json.loads(a.manifest.read_text());groups=m['groups']
def fingerprint(path):
 with h5py.File(path) as f:
  rgb=np.asarray(f['rgb']);depth=np.asarray(f['depth'])
  return hashlib.sha256(rgb.tobytes()).hexdigest(),hashlib.sha256(depth.tobytes()).hexdigest()
train={}; tune={}; eval_rgb={};eval_pair={}
for label,dest in [('train',train),('tune',tune)]:
 for row in groups[label]['samples']:
  path=ROOT/row['path']
  if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('manifest hash mismatch')
  dest[row['path']]=fingerprint(path)
for path in sorted((ROOT/'data/nyu_depth_v2/extracted/val/official').glob('*.h5')):
 rgb,depth=fingerprint(path);eval_rgb.setdefault(rgb,[]).append(path.name);eval_pair.setdefault((rgb,depth),[]).append(path.name)
overlap=[]
for label,group in [('train',train),('tune',tune)]:
 for path,(rgb,depth) in group.items():
  if rgb in eval_rgb:overlap.append({'group':label,'path':path,'evaluation_rgb_matches':eval_rgb[rgb],'paired_matches':eval_pair.get((rgb,depth),[])})
train_tune_rgb=set(x[0] for x in train.values())&set(x[0] for x in tune.values())
scenes={}
for label in ('train','tune'):
 declared=groups[label].get('scenes',[groups[label].get('scene')])
 if not declared or any(not name for name in declared):raise ValueError('Missing scene names')
 actual={Path(row['source_member']).parts[-2] for row in groups[label]['samples']}
 if actual!=set(declared):raise ValueError('Declared and source-member scenes differ')
 scenes[label]=sorted(actual)
report={'scene_names':scenes,'train_tune_scene_disjoint':not (set(scenes['train']) & set(scenes['tune'])),
 'counts':{'train':len(train),'tune':len(tune),'evaluation':sum(len(v) for v in eval_rgb.values())},
 'exact_rgb_overlap_train_tune_count':len(train_tune_rgb),'exact_rgb_overlap_with_evaluation':overlap,
 'limitation':'Exact decoded RGB/depth matches only; near-duplicates, camera-frame similarity and evaluation scene identities remain unverified.',
 'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest()}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if not report['train_tune_scene_disjoint'] or train_tune_rgb or overlap:raise SystemExit(1)
