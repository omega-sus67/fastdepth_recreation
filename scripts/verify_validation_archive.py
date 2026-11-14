#!/usr/bin/env python3
"""Compare validation HDF5 bytes against both local source archives."""
import hashlib,json,tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
archives=sorted((root/'data/nyu_depth_v2/raw').glob('val-*.tar'))
expected=root/'data/nyu_depth_v2/extracted/val/official'
out=root/'results/data/2026-11-13-source-verification/archive-parity.json'
if out.exists():raise SystemExit('Output exists')
if len(archives)!=2:raise SystemExit('Expected two validation archives')
seen=set();errors=[];counts={}
for archive in archives:
 count=0
 with tarfile.open(archive) as tar:
  for m in tar:
   if not m.isfile() or not m.name.endswith('.h5'):continue
   if Path(m.name).parts[:2]!=('val','official') or len(Path(m.name).parts)!=3:
    errors.append({'member':m.name,'reason':'unexpected path'});continue
   name=Path(m.name).name
   if name in seen:errors.append({'member':m.name,'reason':'duplicate name'});continue
   seen.add(name);target=expected/name
   if not target.is_file():errors.append({'member':m.name,'reason':'missing extracted file'});continue
   stream=tar.extractfile(m);digest=hashlib.sha256()
   while chunk:=stream.read(1024*1024):digest.update(chunk)
   local=hashlib.sha256()
   with target.open('rb') as f:
    while chunk:=f.read(1024*1024):local.update(chunk)
   if local.digest()!=digest.digest():errors.append({'member':m.name,'reason':'hash mismatch'})
   count+=1
 counts[archive.name]=count
local_names={p.name for p in expected.glob('*.h5')}
report={'scope':'Byte comparison to local downloaded Hugging Face archives, not independently to original MIT archive',
 'source':'https://huggingface.co/datasets/sayakpaul/nyu_depth_v2','revision':'50579a0b591445f91bf7269b28611ad1cc6a05d2',
 'archive_counts':counts,'member_count':len(seen),'local_count':len(local_names),'extra_local':sorted(local_names-seen),
 'missing_local':sorted(seen-local_names),'errors':errors}
out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if errors or local_names!=seen:raise SystemExit(1)
