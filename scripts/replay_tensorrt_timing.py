#!/usr/bin/env python3
"""Repeat synchronized desktop TensorRT timing using already-built engines."""
import argparse,csv,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,torch,tensorrt as trt
from fastdepth_project.models import sha256

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error('output exists')
 cfg=json.loads(a.config.read_text());engine_config=ROOT/cfg['engine_config'];sources=json.loads(engine_config.read_text())
 logger=trt.Logger(trt.Logger.WARNING);runtime=trt.Runtime(logger);models={}
 for item in sources['models']:
  path=ROOT/item['engine'];engine=runtime.deserialize_cuda_engine(path.read_bytes())
  if engine is None:raise ValueError(f'Cannot load engine {path}')
  context=engine.create_execution_context();x=torch.zeros((1,3,224,224),device='cuda');y=torch.empty((1,1,224,224),device='cuda')
  if not context.set_tensor_address('rgb',x.data_ptr()) or not context.set_tensor_address('depth',y.data_ptr()):raise ValueError('Binding failed')
  models[item['name']]={'engine':engine,'context':context,'x':x,'y':y,'engine_sha256':sha256(path)}
 if set(cfg['order'])!=set(models):raise ValueError('Timing order does not match engines')
 a.output.mkdir(parents=True);stream=torch.cuda.current_stream().cuda_stream;allrows=[];runs=[]
 for number,name in enumerate(cfg['order'],1):
  model=models[name]
  def infer():
   if not model['context'].execute_async_v3(stream):raise RuntimeError('TensorRT inference failed')
  for _ in range(cfg['warmups']):infer()
  torch.cuda.synchronize();values=[]
  for i in range(cfg['calls_per_run']):
   torch.cuda.synchronize();start=time.perf_counter_ns();infer();torch.cuda.synchronize();elapsed=(time.perf_counter_ns()-start)/1e6
   values.append(elapsed);allrows.append({'run':number,'model':name,'iteration':i,'latency_ms':elapsed})
  runs.append({'run':number,'model':name,'engine_sha256':model['engine_sha256'],'median_ms':float(np.median(values)),'p95_ms':float(np.percentile(values,95))})
  print(f"{number}/{len(cfg['order'])} {name}: {runs[-1]['median_ms']:.6f} ms",flush=True)
 with (a.output/'latencies_ms.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['run','model','iteration','latency_ms']);w.writeheader();w.writerows(allrows)
 report={'scope':cfg['timing_scope'],'config_sha256':sha256(a.config),'engine_config_sha256':sha256(engine_config),
  'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__),'tensorrt':trt.__version__,
  'warmups':cfg['warmups'],'calls_per_run':cfg['calls_per_run'],'runs':runs,'limitations':cfg['limitations']}
 (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
