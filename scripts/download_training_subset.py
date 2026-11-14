#!/usr/bin/env python3
"""Extract a bounded preliminary scene-disjoint subset from the first NYU training shard."""
import argparse,hashlib,json,tarfile,urllib.request
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--per-scene',type=int,default=16);p.add_argument('--max-mib',type=int,default=768)
a=p.parse_args()
if a.output.exists():p.error('output exists')
if a.per_scene<1 or a.max_mib<1:p.error('positive limits required')
a.output.mkdir(parents=True)
url='https://huggingface.co/datasets/sayakpaul/nyu_depth_v2/resolve/main/data/train-000000.tar'
class Bounded:
 def __init__(self,stream,limit):self.stream=stream;self.limit=limit;self.n=0
 def read(self,n=-1):
  if self.n>=self.limit:raise EOFError('network byte budget reached')
  chunk=self.stream.read(min(n if n>=0 else 16384,self.limit-self.n));self.n+=len(chunk);return chunk
scenes={};reason='';seen=0
try:
 with urllib.request.urlopen(url,timeout=45) as response:
  stream=Bounded(response,a.max_mib*1024**2)
  with tarfile.open(fileobj=stream,mode='r|') as tar:
   for member in tar:
    parts=Path(member.name).parts
    if not member.isfile() or len(parts)!=3 or parts[0]!='train' or parts[-1] in ('.','..') or not parts[-1].endswith('.h5') or any(x in ('..','.') for x in parts):continue
    scene=parts[1]
    if scene not in scenes and len(scenes)>=2:break
    scenes.setdefault(scene,[])
    if len(scenes[scene])<a.per_scene:
     content=tar.extractfile(member).read();target=a.output/scene/parts[-1];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
     scenes[scene].append({'path':str(target),'sha256':hashlib.sha256(content).hexdigest(),'source_member':member.name})
    if len(scenes)==2 and all(len(v)>=a.per_scene for v in scenes.values()):break
  seen=stream.n
except (OSError,EOFError,tarfile.TarError) as e:reason=str(e);seen=locals().get('stream').n if 'stream' in locals() else 0
report={'source_url':url,'network_bytes_read':seen,'per_scene_requested':a.per_scene,'scenes':scenes,
 'scope':'Preliminary scene-grouped subset; original evaluation scene identity and overlap remain unverified; not for final claims',
 'complete':len(scenes)==2 and all(len(v)>=a.per_scene for v in scenes.values()),'stop_reason':reason}
(a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'scene_counts':{k:len(v) for k,v in scenes.items()},'network_bytes_read':seen,'complete':report['complete'],'stop_reason':reason},indent=2))
