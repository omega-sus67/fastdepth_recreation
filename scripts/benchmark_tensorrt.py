#!/usr/bin/env python3
"""Build a desktop TensorRT engine and measure synchronized model-only latency."""
import argparse,csv,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,torch,tensorrt as trt
from fastdepth_project.data import load_h5,preprocess
from fastdepth_project.models import load_converted,sha256
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--onnx',type=Path,default=ROOT/'exports/2026-11-13-community-pruned/model.onnx')
p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/converted/community-skipadd-pruned.pth')
p.add_argument('--sample',type=Path,default=ROOT/'data/nyu_depth_v2/extracted/val/official/00001.h5')
p.add_argument('--precision',choices=['fp32','fp16'],default='fp16')
p.add_argument('--warmup',type=int,default=50);p.add_argument('--runs',type=int,default=500)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.warmup<0 or a.runs<1:p.error('invalid iteration counts')
if a.output.exists():p.error('output exists')
a.output.mkdir(parents=True)
logger=trt.Logger(trt.Logger.WARNING);builder=trt.Builder(logger)
network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH));parser=trt.OnnxParser(network,logger)
if not parser.parse_from_file(str(a.onnx)):
 raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
config=builder.create_builder_config();config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,512*1024**2)
if a.precision=='fp16':config.set_flag(trt.BuilderFlag.FP16)
serialized=builder.build_serialized_network(network,config)
if serialized is None:raise RuntimeError('TensorRT engine build failed')
enginepath=a.output/'desktop.engine';enginepath.write_bytes(bytes(serialized));runtime=trt.Runtime(logger)
engine=runtime.deserialize_cuda_engine(serialized);context=engine.create_execution_context()
x=torch.zeros((1,3,224,224),device='cuda',dtype=torch.float32);y=torch.empty((1,1,224,224),device='cuda',dtype=torch.float32)
if not context.set_tensor_address('rgb',x.data_ptr()) or not context.set_tensor_address('depth',y.data_ptr()):raise RuntimeError('tensor binding failed')
stream=torch.cuda.current_stream().cuda_stream
def infer():
 if not context.execute_async_v3(stream):raise RuntimeError('TensorRT execution failed')
for _ in range(a.warmup):infer()
torch.cuda.synchronize();times=[]
for _ in range(a.runs):
 torch.cuda.synchronize();start=time.perf_counter_ns();infer();torch.cuda.synchronize();times.append((time.perf_counter_ns()-start)/1e6)
with (a.output/'latencies_ms.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['iteration','latency_ms']);w.writerows(enumerate(times))
rgb,depth=load_h5(a.sample);sample,_=preprocess(rgb,depth);x.copy_(sample.unsqueeze(0));infer();torch.cuda.synchronize();actual=y.cpu().numpy()
model,record=load_converted(a.checkpoint,'cuda')
with torch.inference_mode():expected=model(x).cpu().numpy()
diff=np.abs(actual-expected)
report={'scope':'Desktop TensorRT model-only; preallocated zero tensor; excludes preprocessing and transfers',
 'precision':a.precision,'precision_note':('FP16 tactics enabled; other operations may remain FP32' if a.precision=='fp16' else 'FP16 flag not enabled; TensorRT default TF32 setting was not recorded'),
 'gpu':torch.cuda.get_device_name(0),'tensorrt':trt.__version__,'torch':str(torch.__version__),
 'checkpoint_sha256':sha256(a.checkpoint),'onnx_sha256':sha256(a.onnx),'engine_sha256':sha256(enginepath),
 'warmup':a.warmup,'samples':a.runs,'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),
 'p99_ms':float(np.percentile(times,99)),'sample':a.sample.name,'sample_path':str(a.sample),'sample_max_abs_diff_vs_pytorch':float(diff.max()),
 'sample_mean_abs_diff_vs_pytorch':float(diff.mean()),'model_source':record['provenance']['source_status'],
 'limitations':'One desktop run and one parity sample; not Orin performance, full-set accuracy or end-to-end 33 ms target.'}
(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
