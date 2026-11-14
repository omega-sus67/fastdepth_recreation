#!/usr/bin/env python3
"""Create fixed-width random, weight and two ablation-ranked candidates."""
import argparse,json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from fastdepth_project.models import load_converted,module_spec,sha256
from fastdepth_project.pruning import prune_first_decoder

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--config',type=Path,required=True);p.add_argument('--scores',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error('output exists')
 cfg=json.loads(a.config.read_text());scores=json.loads(a.scores.read_text());source=ROOT/cfg['source_checkpoint']
 if scores['smoke'] or scores['config_sha256']!=sha256(a.config) or scores['checkpoint_sha256']!=sha256(source) or scores['manifest_sha256']!=sha256(ROOT/cfg['manifest']):raise ValueError('Scores do not match protocol')
 if sorted(v['channel'] for v in scores['channels'])!=list(range(200)):raise ValueError('Incomplete channel scores')
 width=cfg['pruning']['width'];remove=200-width
 if not 1<=width<=200:raise ValueError('Unsupported width')
 selections={'weight':None}
 for name,key in [('average_ablation','mean_delta'),('scene_aware','scene_aware_score')]:
  selections[name]=[r['channel'] for r in sorted(scores['channels'],key=lambda r:(r[key],r['channel']))[:remove]]
 for seed in cfg['pruning']['random_seeds']:
  selections[f'random_{seed}']=random.Random(seed).sample(range(200),remove)
 model,record=load_converted(source,'cpu');a.output.mkdir(parents=True);results={}
 for name,removed in selections.items():
  candidate,details=prune_first_decoder(model,width,removed)
  details.update({'method':name,'config_sha256':sha256(a.config),'score_sha256':sha256(a.scores) if name in ('average_ablation','scene_aware') else None})
  folder=a.output/name;folder.mkdir();path=folder/'candidate.pth'
  torch.save({'schema_version':1,'variant':record['variant']+'-scene-experiment-'+name,'model_config':module_spec(candidate),
    'state_dict':candidate.state_dict(),'provenance':{'source_status':'project_pruning_candidate',
    'source_checkpoint_sha256':sha256(source),'inherited_checkpoint_source':record['provenance']['source_status'],'pruning':details}},path)
  restored,_=load_converted(path,'cpu');x=torch.zeros(1,3,32,32)
  with torch.inference_mode():
   y=candidate(x);torch.testing.assert_close(y,restored(x),atol=0,rtol=0)
   if not bool(torch.isfinite(y).all()):raise ValueError('Nonfinite output')
  results[name]={'path':str(path.resolve().relative_to(ROOT)),'sha256':sha256(path),
    'parameters':sum(p.numel() for p in candidate.parameters()),'pruning':details,'reload_exact':True}
 (a.output/'candidates.json').write_text(json.dumps({'config_sha256':sha256(a.config),'source_sha256':sha256(source),'models':results},indent=2)+'\n')
 print(json.dumps({n:{'removed':v['pruning']['removed_channels'],'parameters':v['parameters']} for n,v in results.items()},indent=2))

if __name__=='__main__':main()
