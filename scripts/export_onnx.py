#!/usr/bin/env python3
"""Export the recovered model and compare on a real validation input on CPU."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,onnx,onnxruntime as ort,torch
from fastdepth_project.data import load_h5,preprocess
from fastdepth_project.models import load_converted,sha256
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/converted/community-skipadd-pruned.pth')
p.add_argument('--sample',type=Path,default=ROOT/'data/nyu_depth_v2/extracted/val/official/00001.h5')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():p.error('output exists')
a.output.mkdir(parents=True)
model,record=load_converted(a.checkpoint,'cpu'); rgb,depth=load_h5(a.sample);x,_=preprocess(rgb,depth);x=x.unsqueeze(0)
with torch.inference_mode(): expected=model(x).numpy()
path=a.output/'model.onnx'
torch.onnx.export(model,(x,),str(path),input_names=['rgb'],output_names=['depth'],opset_version=18,dynamo=True,external_data=False)
onnx.checker.check_model(onnx.load(path)); session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
actual=session.run(None,{'rgb':x.numpy()})[0]; difference=np.abs(actual-expected)
report={'scope':'One supplied HDF5 input; CPU PyTorch versus ONNX Runtime CPU; not Jetson validation',
 'sample':a.sample.name,'sample_path':str(a.sample),'checkpoint_sha256':sha256(a.checkpoint),'onnx_sha256':sha256(path),
 'model_source':record['provenance']['source_status'],'opset':18,'output_shape':list(actual.shape),
 'max_abs_error':float(difference.max()),'mean_abs_error':float(difference.mean()),'tolerance':1e-4,
 'within_tolerance':bool(difference.max()<=1e-4),'onnxruntime':ort.__version__,'torch':str(torch.__version__)}
(a.output/'parity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if not report['within_tolerance']:raise SystemExit('ONNX parity exceeded tolerance')
