#!/usr/bin/env python3
"""Reconstruct comparison RMSE aggregates from saved per-image records."""
import argparse,collections,csv,hashlib,json,math
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--comparison',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error('output exists')
 summary=a.comparison/'summary.json';images=a.comparison/'per_image.csv';s=json.loads(summary.read_text());data=collections.defaultdict(list)
 with images.open(newline='') as f:
  for row in csv.DictReader(f):data[row['model']].append(row)
 errors=[];maximum=0.
 if set(data)!=set(s['models']):errors.append('model set mismatch')
 for name,model in s['models'].items():
  rows=data[name]
  if len(rows)!=s['samples'] or len({x['sample'] for x in rows})!=s['samples']:errors.append(name+' sample count')
  if not rows:continue
  mean=sum(float(x['rmse']) for x in rows)/len(rows);pixels=sum(int(x['valid_pixels']) for x in rows)
  pooled=math.sqrt(sum(float(x['rmse'])**2*int(x['valid_pixels']) for x in rows)/pixels)
  delta=max(abs(mean-model['mean_image_rmse']),abs(pooled-model['pooled']['rmse']));maximum=max(maximum,delta)
  if delta>1e-10:errors.append(name+' RMSE aggregation')
  scenes=collections.defaultdict(list)
  for row in rows:scenes[row['scene']].append(float(row['rmse']))
  if set(scenes)!=set(model['scenes']):errors.append(name+' scene set')
  for scene,values in scenes.items():
   if abs(sum(values)/len(values)-model['scenes'][scene]['mean_image_rmse'])>1e-10:errors.append(name+' '+scene)
 out={'scope':'Reconstruct aggregate RMSE from per-image records; not a proof of model correctness or independent test data',
  'summary_sha256':sha(summary),'per_image_sha256':sha(images),'models':len(data),'rows':sum(map(len,data.values())),
  'max_absolute_difference':maximum,'errors':errors}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
 if errors:raise SystemExit(1)

if __name__=='__main__':main()
