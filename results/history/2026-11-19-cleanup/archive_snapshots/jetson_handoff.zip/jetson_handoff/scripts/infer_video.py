#!/usr/bin/env python3
"""Write a colorized depth video from a recorded input; no audio is retained."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import cv2,numpy as np,torch
from fastdepth_project.data import preprocess
from fastdepth_project.models import load_converted,sha256
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('video',type=Path);p.add_argument('--output',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/converted/community-skipadd-pruned.pth')
p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');a=p.parse_args()
if a.output.exists():p.error('output exists')
cap=cv2.VideoCapture(str(a.video))
if not cap.isOpened():p.error('cannot open input video')
fps=cap.get(cv2.CAP_PROP_FPS)
if not np.isfinite(fps) or fps<=0:p.error('input video has no valid FPS')
model,record=load_converted(a.checkpoint,a.device);a.output.mkdir(parents=True)
writer=cv2.VideoWriter(str(a.output/'depth.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(224,224))
if not writer.isOpened():raise RuntimeError('cannot open video writer')
count=0;start=time.perf_counter()
try:
 while True:
  ok,frame=cap.read()
  if not ok:break
  rgb=cv2.cvtColor(cv2.resize(frame,(640,480),interpolation=cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB)
  x=preprocess(rgb)
  with torch.inference_mode():prediction=model(x.unsqueeze(0).to(a.device)).cpu().numpy()[0,0]
  if not np.isfinite(prediction).all():raise ValueError('nonfinite prediction')
  preview=cv2.normalize(prediction,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
  writer.write(cv2.applyColorMap(preview,cv2.COLORMAP_MAGMA));count+=1
finally:cap.release();writer.release()
elapsed=time.perf_counter()-start
if not count:raise ValueError('input video contained no readable frames')
report={'frames':count,'elapsed_seconds':elapsed,'complete_pipeline_fps':count/elapsed,'input_fps':fps,
 'input_resize':'all frames resized to 640x480; aspect ratio may change','preview':'per-frame normalized color; not metric depth',
 'checkpoint_sha256':sha256(a.checkpoint),'model_source':record['provenance']['source_status'],'device':a.device,
 'timing_scope':'decode, resize, preprocess, transfer, inference, colorize and encode; no audio'}
(a.output/'metadata.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
