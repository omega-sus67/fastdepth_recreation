#!/usr/bin/env python3
"""Run a converted FastDepth checkpoint on a 480x640 RGB image."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,torch
from PIL import Image
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fastdepth_project.data import preprocess
from fastdepth_project.models import load_converted,sha256
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('image',type=Path);p.add_argument('--output',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/converted/community-skipadd-pruned.pth')
p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');a=p.parse_args()
if a.output.exists():p.error('output exists')
try:
 with Image.open(a.image) as im:rgb=np.asarray(im.convert('RGB'))
except (OSError,ValueError) as e:p.error(f'cannot read image: {e}')
try:x=preprocess(rgb)
except ValueError as e:p.error(str(e))
model,record=load_converted(a.checkpoint,a.device)
with torch.inference_mode():depth=model(x.unsqueeze(0).to(a.device)).cpu().numpy()[0,0]
if not np.isfinite(depth).all():raise ValueError('nonfinite prediction')
a.output.mkdir(parents=True);np.save(a.output/'depth.npy',depth)
fig,ax=plt.subplots(figsize=(5,5));im=ax.imshow(depth,cmap='magma');ax.axis('off');fig.colorbar(im,ax=ax,label='Predicted depth (units unverified)');fig.tight_layout();fig.savefig(a.output/'preview.png',dpi=130);plt.close(fig)
(a.output/'metadata.json').write_text(json.dumps({'image':str(a.image),'checkpoint_sha256':sha256(a.checkpoint),'model_source':record['provenance']['source_status'],'device':a.device,'shape':list(depth.shape),'unit':'unverified','preview_scaling':'automatic per image'},indent=2)+'\n')
print(f'Saved numerical depth and preview to {a.output}')
