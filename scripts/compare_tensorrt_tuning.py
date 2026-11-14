#!/usr/bin/env python3
"""Compare locally built TensorRT engines with PyTorch on tuning images only."""
import argparse,csv,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,torch,tensorrt as trt
from fastdepth_project.data import load_h5,preprocess
from fastdepth_project.metrics import DepthMetrics
from fastdepth_project.models import load_converted,sha256

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error('output exists')
 cfg=json.loads(a.config.read_text());manifest=ROOT/cfg['manifest'];overlap=ROOT/cfg['overlap_report'];m=json.loads(manifest.read_text());o=json.loads(overlap.read_text())
 if o['manifest_sha256']!=sha256(manifest) or not o['train_tune_scene_disjoint'] or o['exact_rgb_overlap_with_evaluation'] or o['exact_rgb_overlap_train_tune_count']:raise ValueError('Overlap audit failed')
 samples=m['groups']['tune']['samples']
 if not samples:raise ValueError('Empty tuning set')
 logger=trt.Logger(trt.Logger.ERROR);runtime=trt.Runtime(logger);models={}
 for item in cfg['models']:
  name=item['name'];engine_path=ROOT/item['engine'];checkpoint=ROOT/item['checkpoint']
  if name in models:raise ValueError('Duplicate model name')
  engine=runtime.deserialize_cuda_engine(engine_path.read_bytes())
  if engine is None:raise ValueError(f'Cannot deserialize {engine_path}')
  model,_=load_converted(checkpoint,'cuda');model.eval();context=engine.create_execution_context()
  x=torch.empty((1,3,224,224),device='cuda');y=torch.empty((1,1,224,224),device='cuda')
  if not context.set_tensor_address('rgb',x.data_ptr()) or not context.set_tensor_address('depth',y.data_ptr()):raise ValueError('Binding failed')
  models[name]={'engine':engine,'context':context,'x':x,'y':y,'model':model,'checkpoint_sha256':sha256(checkpoint),
   'engine_sha256':sha256(engine_path),'trt':DepthMetrics(),'pytorch':DepthMetrics(),'trt_rmse':[],'pytorch_rmse':[],'abs_sum':0.,'abs_count':0,'max_abs':0.}
 stream=torch.cuda.current_stream().cuda_stream;rows=[];a.output.mkdir(parents=True);start=time.monotonic()
 with torch.inference_mode():
  for i,row in enumerate(samples,1):
   path=ROOT/row['path']
   if sha256(path)!=row['sha256']:raise ValueError(f'Sample hash mismatch: {path}')
   rgb,depth=load_h5(path);x,y=preprocess(rgb,depth);target=y.numpy()[0]
   for name,entry in models.items():
    entry['x'].copy_(x.unsqueeze(0));expected=entry['model'](entry['x'])
    if not entry['context'].execute_async_v3(stream):raise RuntimeError('TensorRT inference failed')
    torch.cuda.synchronize();pyt=expected.cpu().numpy()[0,0];pred=entry['y'].cpu().numpy()[0,0]
    difference=np.abs(pred.astype(np.float64)-pyt.astype(np.float64))
    if not np.isfinite(difference).all():raise ValueError('Nonfinite output')
    entry['abs_sum']+=float(difference.sum());entry['abs_count']+=difference.size;entry['max_abs']=max(entry['max_abs'],float(difference.max()))
    for label,value in [('trt',pred),('pytorch',pyt)]:
     one=DepthMetrics();one.update(value,target);v=one.result();entry[label].update(value,target);entry[label+'_rmse'].append(v['rmse'])
     rows.append({'model':name,'runtime':label,'sample':row['path'],'scene':Path(row['source_member']).parts[-2],
                  'output_mean_abs_difference':float(difference.mean()),'output_max_abs_difference':float(difference.max()),
                  'output_pixels_abs_difference_over_0_1':int((difference>0.1).sum()),**v})
   if i%50==0 or i==len(samples):print(f'{i}/{len(samples)} tuning images',flush=True)
 with (a.output/'per_image.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 summary={}
 for name,e in models.items():
  summary[name]={'checkpoint_sha256':e['checkpoint_sha256'],'engine_sha256':e['engine_sha256'],
    'trt_pooled':e['trt'].result(),'trt_mean_image_rmse':float(np.mean(e['trt_rmse'])),
    'pytorch_pooled':e['pytorch'].result(),'pytorch_mean_image_rmse':float(np.mean(e['pytorch_rmse'])),
    'mean_absolute_output_difference':e['abs_sum']/e['abs_count'],'max_absolute_output_difference':e['max_abs']}
 report={'scope':'Desktop FP16-enabled TensorRT versus PyTorch on 445 tuning images; not final evaluation or Orin',
  'config_sha256':sha256(a.config),'manifest_sha256':sha256(manifest),'overlap_sha256':sha256(overlap),
  'samples':len(samples),'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__),'tensorrt':trt.__version__,
  'precision_note':'FP16 tactics enabled; some engine operations may remain FP32; PyTorch FP32 reference',
  'depth_units':'native dataset units; exact conversion chain unverified','models':summary,
  'elapsed_seconds':time.monotonic()-start,'limitations':cfg['limitations']}
 (a.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:{'trt_mean_image_rmse':v['trt_mean_image_rmse'],'pytorch_mean_image_rmse':v['pytorch_mean_image_rmse'],
                    'max_abs':v['max_absolute_output_difference']} for k,v in summary.items()},indent=2))

if __name__=='__main__':main()
