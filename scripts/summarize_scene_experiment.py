#!/usr/bin/env python3
"""Summarize paired seed and inferred-location results from the fixed comparison."""
import argparse,collections,csv,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
 import hashlib
 return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--comparison',type=Path,required=True)
 p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error('output exists')
 manifest=json.loads(a.manifest.read_text());summary=json.loads((a.comparison/'summary.json').read_text())
 if summary['manifest_sha256']!=sha(a.manifest):raise ValueError('Manifest mismatch')
 sample_groups={r['path']:r['location_group'] for r in manifest['groups']['tune']['samples']}
 data=collections.defaultdict(lambda:collections.defaultdict(list));samples=collections.Counter()
 with (a.comparison/'per_image.csv').open(newline='') as f:
  for row in csv.DictReader(f):
   if row['sample'] not in sample_groups:raise ValueError('Sample absent from tuning manifest')
   data[row['model']][sample_groups[row['sample']]].append(float(row['rmse']));samples[row['model']]+=1
 if set(data)!=set(summary['models']) or any(v!=len(sample_groups) for v in samples.values()):raise ValueError('Incorrect model or sample counts')
 means={name:{group:statistics.mean(values) for group,values in groups.items()} for name,groups in data.items()}
 expected=set(sample_groups.values())
 if any(set(groups)!=expected for groups in means.values()):raise ValueError('Missing location groups')
 comparisons={}
 for method in ['weight','average_ablation','scene_aware','random']:
  seeds=[]
  for seed in [42,43,44]:
   name=f'{method}_{seed}';base=f'original_{seed}';current=means[name];ref=means[base]
   deltas={g:current[g]-ref[g] for g in sorted(expected)}
   worst=max(current,key=current.get);worst_delta=max(deltas,key=deltas.get)
   row={'seed':seed,'mean_image_rmse':summary['models'][name]['mean_image_rmse'],
    'pooled_rmse':summary['models'][name]['pooled']['rmse'],'pooled_delta1':summary['models'][name]['pooled']['delta1'],
    'mean_location_group_rmse':statistics.mean(current.values()),'worst_group':worst,'worst_group_rmse':current[worst],
    'largest_increase_vs_matched_original_group':worst_delta,'largest_increase_vs_matched_original_rmse':deltas[worst_delta],
    'groups_beating_matched_original':sum(v<0 for v in deltas.values()),'per_group_delta_vs_matched_original':deltas}
   if method=='scene_aware':
    comp=means[f'average_ablation_{seed}'];row['groups_beating_average_ablation']=sum(current[g]<comp[g] for g in expected)
    row['mean_group_delta_vs_average_ablation']=statistics.mean(current[g]-comp[g] for g in expected)
   seeds.append(row)
  comparisons[method]={'per_seed':seeds,'mean_of_seed_mean_image_rmse':statistics.mean(s['mean_image_rmse'] for s in seeds),
   'seed_mean_image_rmse_range':[min(s['mean_image_rmse'] for s in seeds),max(s['mean_image_rmse'] for s in seeds)]}
 out={'scope':'Descriptive paired results on previously inspected tuning set, not a final test or Orin measurement',
  'comparison_summary_sha256':sha(a.comparison/'summary.json'),'per_image_sha256':sha(a.comparison/'per_image.csv'),
  'manifest_sha256':sha(a.manifest),'images':len(sample_groups),'inferred_location_groups':len(expected),
  'group_aggregation':'Per inferred suffix-normalized location group, average per-image RMSE, then equally weight groups',
  'limitations':'Frames correlated; groups inferred by name; random mask and order both vary by seed; checkpoint pretraining unknown; no confidence interval or hypothesis test.',
  'comparisons':comparisons}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:{'mean_rmse':v['mean_of_seed_mean_image_rmse'],'range':v['seed_mean_image_rmse_range'],
  'worst_group_rmse':[s['worst_group_rmse'] for s in v['per_seed']]} for k,v in comparisons.items()},indent=2))

if __name__=='__main__':main()
