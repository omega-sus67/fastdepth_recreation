#!/usr/bin/env python3
"""Create one preregistered final-decoder channel-pruning candidate."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from fastdepth_project.models import load_converted,module_spec,sha256
from fastdepth_project.pruning import prune_first_decoder, prune_last_decoder
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/converted/community-skipadd-pruned.pth')
p.add_argument('--stage',choices=['first','last'],default='last')
p.add_argument('--ablation-report',type=Path,help='First-stage single-channel ablation scores, lowest loss increase removed first')
p.add_argument('--keep',type=int,default=12);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():p.error('output exists')
model,record=load_converted(a.checkpoint,'cpu')
if a.ablation_report:
 if a.stage!='first':p.error('ablation report only supports first stage')
 ablation=json.loads(a.ablation_report.read_text())
 if ablation['checkpoint_sha256']!=sha256(a.checkpoint):raise ValueError('Ablation checkpoint differs')
 scores=ablation['channels']
 if len(scores)!=200 or sorted(row['channel'] for row in scores)!=list(range(200)):raise ValueError('Incomplete ablation scores')
 removed=[row['channel'] for row in sorted(scores,key=lambda row:(row['loss_delta'],row['channel']))[:200-a.keep]]
 candidate,details=prune_first_decoder(model,a.keep,removed)
 details['ablation_report_sha256']=sha256(a.ablation_report)
else:
 candidate,details=(prune_first_decoder if a.stage=='first' else prune_last_decoder)(model,a.keep)
torch.manual_seed(42);x=torch.rand(1,3,224,224)
with torch.inference_mode():y=candidate(x)
if list(y.shape)!=[1,1,224,224] or not torch.isfinite(y).all():raise ValueError('invalid candidate output')
a.output.mkdir(parents=True)
output=a.output/'candidate.pth'
torch.save({'schema_version':1,'variant':record['variant']+f'-{a.stage}decoder-pruned',
            'model_config':module_spec(candidate),'state_dict':candidate.state_dict(),
            'provenance':{'source_status':'project_pruning_candidate','source_checkpoint_sha256':sha256(a.checkpoint),
                          'inherited_checkpoint_source':record['provenance']['source_status'],'pruning':details}},output)
restored,_=load_converted(output,'cpu')
with torch.inference_mode():torch.testing.assert_close(restored(x),y,rtol=0,atol=0)
report={'scope':'one fixed structural candidate; not yet fine-tuned or Orin-optimized','source_sha256':sha256(a.checkpoint),
        'candidate_sha256':sha256(output),'pruning':details,'base_parameters':sum(p.numel() for p in model.parameters()),
        'candidate_parameters':sum(p.numel() for p in candidate.parameters()),'output_shape':list(y.shape),
        'finite':True,'reload_bit_exact':True,'accuracy_measured':False,'orin_latency_measured':False}
(a.output/'pruning.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
