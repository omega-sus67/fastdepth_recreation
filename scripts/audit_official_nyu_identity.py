#!/usr/bin/env python3
"""Bounded remote NYU labeled-image exact-match check using HTTP byte ranges."""
import argparse,collections,hashlib,io,json,urllib.request
from pathlib import Path
import h5py,numpy as np
ROOT=Path(__file__).resolve().parents[1]
URL='https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'

class RangeFile(io.RawIOBase):
 def __init__(self,url,size,block=4*1024*1024,max_blocks=32):
  self.url=url;self.size=size;self.block=block;self.max_blocks=max_blocks;self.pos=0;self.cache=collections.OrderedDict();self.bytes_fetched=0;self.requests=0
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
  if self.pos<0:raise ValueError('negative seek')
  return self.pos
 def readinto(self,target):
  dest=memoryview(target);done=0
  while done<len(dest) and self.pos<self.size:
   index=self.pos//self.block
   if index not in self.cache:
    start=index*self.block;end=min(self.size-1,start+self.block-1)
    req=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end}'})
    with urllib.request.urlopen(req,timeout=45) as response:
     if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{self.size}':raise ValueError('Unexpected HTTP range response')
     content=response.read(end-start+2)
    if len(content)!=end-start+1:raise ValueError('Short HTTP range')
    self.requests+=1;self.bytes_fetched+=len(content);self.cache[index]=content
    if len(self.cache)>self.max_blocks:self.cache.popitem(last=False)
   self.cache.move_to_end(index);chunk=self.cache[index];offset=self.pos%self.block;n=min(len(dest)-done,len(chunk)-offset)
   dest[done:done+n]=chunk[offset:offset+n];done+=n;self.pos+=n
  return done

def digest(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=32)
 p.add_argument('--local-indices',action='store_true',help='Sample evenly spaced evaluation filenames as proposed one-based official indices')
 a=p.parse_args()
 if a.output.exists():p.error('output exists')
 if a.limit<1 or a.limit>1449:p.error('limit must be between 1 and 1449')
 evaluation={}
 paths=sorted((ROOT/'data/nyu_depth_v2/extracted/val/official').glob('*.h5'))
 for path in paths:
  with h5py.File(path) as f:rgb=np.asarray(f['rgb'])
  evaluation.setdefault(digest(rgb),[]).append(path.name)
 if not evaluation:raise ValueError('Missing local evaluation set')
 request=urllib.request.Request(URL,method='HEAD')
 with urllib.request.urlopen(request,timeout=30) as response:size=int(response.headers['Content-Length'])
 if a.local_indices:
  if a.limit>len(paths):raise ValueError('More requested indices than local evaluation images')
  positions=[round(i*(len(paths)-1)/(a.limit-1)) for i in range(a.limit)] if a.limit>1 else [0]
  indices=[int(paths[i].stem)-1 for i in positions]
 else:indices=list(range(a.limit))
 remote=RangeFile(URL,size);matches=[];checked=[]
 with h5py.File(remote,'r') as f:
  images=f['images']
  if images.shape!=(1449,3,640,480):raise ValueError(f'Unexpected official image shape {images.shape}')
  for i,index in enumerate(indices,1):
   # MATLAB v7.3 stores width before height; transpose to the project's CHW convention.
   rgb=np.asarray(images[index]).transpose(0,2,1);sha=digest(rgb);hits=evaluation.get(sha,[])
   checked.append({'zero_based_official_index':index,'rgb_sha256':sha})
   if hits:
    ref=f['scenes'][0,index];scene=''.join(chr(int(c)) for c in np.asarray(f[ref]).ravel())
    matches.append({'zero_based_official_index':index,'scene':scene,'local_evaluation_files':hits,'rgb_sha256':sha})
   if i%8==0:print(f'Checked {i}/{a.limit} original labeled images',flush=True)
 report={'scope':'Bounded exact RGB matching; not full original labeled file verification','official_url':URL,'official_bytes':size,
  'local_evaluation_files':sum(map(len,evaluation.values())),'official_images_total':1449,'checked_first_n':a.limit if not a.local_indices else None,
  'selection':'evenly spaced local evaluation filenames interpreted as one-based official indices' if a.local_indices else 'first original labeled images',
  'orientation':'official 3x640x480 transposed to 3x480x640','matches':matches,'checked':checked,
  'remote_requests':remote.requests,'remote_bytes_fetched':remote.bytes_fetched,
  'limitation':'Checked images are a subset, not a complete mapping; no near-duplicate or checkpoint-training provenance verification.'}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k!='checked'},indent=2))

if __name__=='__main__':main()
