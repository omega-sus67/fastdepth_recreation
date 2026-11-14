#!/usr/bin/env python3
"""Flag visually similar scene-split/evaluation images using a small difference hash."""
import argparse, hashlib, json
from pathlib import Path
import h5py
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]

def difference_hash(rgb):
    """Convert CHW RGB to a 64-bit horizontal luminance-difference hash."""
    if rgb.shape != (3,480,640):raise ValueError(f'Unexpected RGB shape {rgb.shape}')
    image=Image.fromarray(np.moveaxis(rgb,0,2),'RGB').convert('L').resize((9,8),Image.Resampling.LANCZOS)
    a=np.asarray(image,dtype=np.uint8)
    bits=(a[:,1:]>a[:,:-1]).flatten()
    return sum(int(bit)<<i for i,bit in enumerate(bits))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threshold',type=int,default=3);a=p.parse_args()
    if a.output.exists():p.error('output exists')
    if not 0<=a.threshold<=64:p.error('threshold must be 0 to 64')
    m=json.loads(a.manifest.read_text());groups={}
    for label in ('train','tune'):
        groups[label]=[]
        for item in m['groups'][label]['samples']:
            path=ROOT/item['path']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Manifest hash mismatch')
            with h5py.File(path) as f:value=difference_hash(np.asarray(f['rgb']))
            groups[label].append((item['path'],value))
    evaluation=[]
    for path in sorted((ROOT/'data/nyu_depth_v2/extracted/val/official').glob('*.h5')):
        with h5py.File(path) as f:evaluation.append((path.name,difference_hash(np.asarray(f['rgb']))))
    if not evaluation:raise ValueError('Missing evaluation files')
    pairs={}
    for name,left,right in [('train_tune',groups['train'],groups['tune']),('train_evaluation',groups['train'],evaluation),('tune_evaluation',groups['tune'],evaluation)]:
        count=0;examples=[];minimum=64
        for lp,lh in left:
            for rp,rh in right:
                distance=(lh^rh).bit_count();minimum=min(minimum,distance)
                if distance<=a.threshold:
                    count+=1
                    if len(examples)<30:examples.append({'left':lp,'right':rp,'distance':distance})
        pairs[name]={'pairs_checked':len(left)*len(right),'flagged_pairs':count,'minimum_distance':minimum if left and right else None,'examples_first_30':examples}
    report={'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),'hash':'64-bit horizontal dHash after grayscale 9x8 LANCZOS resize',
        'threshold_hamming_distance_at_most':a.threshold,'pairs':pairs,
        'limitations':'A low hash distance is only a visual similarity flag, not proof of duplicate; some near duplicates may have larger distances; not a scene identity check.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
