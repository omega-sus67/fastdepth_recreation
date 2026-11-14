#!/usr/bin/env python3
"""Extract small scene-grouped samples from two pinned NYU training shards."""
import argparse,hashlib,io,json,tarfile,time,urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REV='50579a0b591445f91bf7269b28611ad1cc6a05d2'
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True);p.add_argument('--per-scene',type=int,default=16)
p.add_argument('--prefix-mib',type=int,default=24);a=p.parse_args()
if a.output.exists():p.error('output exists')
if a.per_scene<1 or a.prefix_mib<1:p.error('positive limits required')
a.output.mkdir(parents=True)
def get_shard(shard):
 url=f'https://huggingface.co/datasets/sayakpaul/nyu_depth_v2/resolve/{REV}/data/train-{shard:06d}.tar'
 last=a.prefix_mib*1024*1024-1;errors=[]
 for attempt in range(3):
  try:
   req=urllib.request.Request(url,headers={'Range':f'bytes=0-{last}'})
   with urllib.request.urlopen(req,timeout=90) as response:
    cr=response.headers.get('Content-Range','')
    if response.status!=206 or not cr.startswith('bytes 0-'):raise ValueError(f'Expected HTTP 206 starting at zero, got {response.status} {cr}')
    body=response.read(last+1)
   entries=[];scene=None
   with tarfile.open(fileobj=io.BytesIO(body),mode='r:') as tar:
    for member in tar:
     parts=Path(member.name).parts
     if not member.isfile() or len(parts)!=3 or parts[0]!='train' or any(q in ('.','..') for q in parts) or not parts[-1].endswith('.h5'):continue
     if scene is None:scene=parts[1]
     if parts[1]!=scene:break
     if member.size>5*1024*1024:raise ValueError('Unexpectedly large sample')
     data=tar.extractfile(member).read()
     if len(data)!=member.size:raise ValueError('Incomplete archive member')
     entries.append((member.name,data))
     if len(entries)==a.per_scene:break
   if len(entries)!=a.per_scene:raise ValueError(f'Only {len(entries)} complete samples in prefix')
   return {'shard':shard,'url':url,'content_range':cr,'bytes_downloaded':len(body),'scene':scene,'prior_errors':errors},entries
  except Exception as exc:
   errors.append(f'{type(exc).__name__}: {exc}')
   if attempt<2:time.sleep(2*(attempt+1))
 raise RuntimeError(f'shard {shard} failed: {errors}')
try:
 with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(get_shard,[0,1]))
 if results[0][0]['scene']==results[1][0]['scene']:raise ValueError('First scenes are identical across shards')
 manifest={'dataset_revision':REV,'scope':'Small provisional scene-disjoint training/tuning subset; validation scene identities remain unverified','groups':{},'sources':[]}
 for split,(meta,entries) in zip(('train','tune'),results):
  scene=meta['scene'];paths=[]
  for member,data in entries:
   target=a.output/split/scene/Path(member).name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
   paths.append({'path':str(target.resolve().relative_to(ROOT)),'source_member':member,'sha256':hashlib.sha256(data).hexdigest()})
  manifest['groups'][split]={'scene':scene,'samples':paths};manifest['sources'].append(meta)
 manifest['scene_disjoint']=True
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({'scenes':{k:v['scene'] for k,v in manifest['groups'].items()},'samples_per_scene':a.per_scene,'sources':manifest['sources']},indent=2))
except Exception as exc:
 (a.output/'failure.json').write_text(json.dumps({'error':str(exc),'dataset_revision':REV},indent=2)+'\n')
 raise
