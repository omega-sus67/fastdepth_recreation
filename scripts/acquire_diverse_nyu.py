#!/usr/bin/env python3
"""Acquire a bounded, scene-grouped NYU sample from multiple tar-shard ranges."""
import argparse, concurrent.futures, hashlib, io, json, math, random, re, tarfile, time, urllib.request
from pathlib import Path
import h5py

ROOT = Path(__file__).resolve().parents[1]
REV = '50579a0b591445f91bf7269b28611ad1cc6a05d2'
API = f'https://huggingface.co/api/datasets/sayakpaul/nyu_depth_v2/tree/{REV}/data?recursive=false&expand=false'

def parse_window(body, start):
    """Locate a valid tar header on a 512-byte boundary and decode complete h5 members."""
    if start % 512:
        raise ValueError('start must align to 512 bytes')
    found = []
    pos = 0
    while pos + 512 <= len(body):
        if body[pos+257:pos+262] != b'ustar':
            pos += 512
            continue
        try:
            member = tarfile.TarInfo.frombuf(body[pos:pos+512], 'utf-8', 'strict')
        except (tarfile.HeaderError, ValueError, UnicodeError):
            pos += 512
            continue
        end = pos + 512 + member.size
        step = 512 + ((member.size + 511) // 512) * 512
        parts = Path(member.name).parts
        if (member.isfile() and len(parts) == 3 and parts[0] == 'train'
                and all(re.fullmatch(r'[A-Za-z0-9_.-]+', part) and part not in ('.','..') for part in parts)
                and parts[-1].endswith('.h5') and 0 < member.size <= 5 * 1024 * 1024 and end <= len(body)):
            found.append((member.name, body[pos+512:end], start+pos))
        pos += step if step > 0 else 512
    return found

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--windows-per-shard',type=int,default=5)
    p.add_argument('--window-mib',type=int,default=32)
    p.add_argument('--per-window',type=int,default=16)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--exclude-manifest',type=Path,default=ROOT/'data/nyu_depth_v2/multiscene_2026-11-14-expanded/manifest.json')
    a=p.parse_args()
    if a.output.exists():p.error('output exists')
    if min(a.windows_per_shard,a.window_mib,a.per_window,a.workers)<1:p.error('positive sizes required')
    try:a.output.resolve().relative_to(ROOT)
    except ValueError:p.error('output must be within project root')
    exclude=json.loads(a.exclude_manifest.read_text())
    excluded={s for group in exclude['groups'].values() for s in group['scenes']}
    a.output.mkdir(parents=True)
    with urllib.request.urlopen(API,timeout=60) as response: listing=json.load(response)
    shards=sorted((x['path'],x['size']) for x in listing if re.fullmatch(r'data/train-\d{6}\.tar',x['path']))
    if not shards:raise ValueError('No training shards found')
    window=a.window_mib*1024*1024
    requests=[]
    for name,size in shards:
        if size<=window:raise ValueError(f'Shard too small: {name}')
        for i in range(a.windows_per_shard):
            start=(int((size-window)*(i+1)/(a.windows_per_shard+1))//512)*512
            requests.append((name,size,start,start+window-1))
    def fetch(task):
        name,size,start,end=task;errors=[]
        # A distinct query per range avoids a cached signed redirect for a different byte range.
        url=f'https://huggingface.co/datasets/sayakpaul/nyu_depth_v2/resolve/{REV}/{name}?range_start={start}&range_end={end}'
        for attempt in range(3):
            try:
                req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(req,timeout=120) as response:
                    cr=response.headers.get('Content-Range','')
                    if response.status!=206 or cr!=f'bytes {start}-{end}/{size}':raise ValueError(f'Unexpected range: {response.status} {cr}')
                    body=response.read(window+1)
                if len(body)!=window:raise ValueError(f'Expected {window} bytes; got {len(body)}')
                entries=parse_window(body,start)
                entries=[x for x in entries if Path(x[0]).parts[1] not in excluded]
                if not entries:return {'shard':name,'range':cr,'errors':errors,'found':0},[]
                # Even spacing within this bounded window reduces, but does not eliminate, temporal correlation.
                indexes=sorted({round(i*(len(entries)-1)/(min(a.per_window,len(entries))-1)) for i in range(min(a.per_window,len(entries)))}) if len(entries)>1 and a.per_window>1 else [0]
                selected=[]
                for index in indexes:
                    member,data,header=entries[index]
                    with h5py.File(io.BytesIO(data),'r') as f:
                        if f['rgb'].shape!=(3,480,640) or f['depth'].shape!=(480,640):raise ValueError(f'Unexpected HDF5 shape {member}')
                    selected.append((member,data,header))
                return {'shard':name,'range':cr,'errors':errors,'found':len(entries),'selected':len(selected)},selected
            except Exception as exc:
                errors.append(f'{type(exc).__name__}: {exc}')
                if attempt<2:time.sleep((attempt+1)*2)
        return {'shard':name,'start':start,'end':end,'errors':errors,'failed':True},[]
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i,result in enumerate(pool.map(fetch,requests),1):
            results.append(result)
            print(f'Processed {i}/{len(requests)} ranges',flush=True)
    groups_by_scene={}
    for metadata,entries in results:
        for member,data,header in entries:
            scene=Path(member).parts[1]
            groups_by_scene.setdefault(scene,{})[member]=(data,metadata['shard'],header)
    if len(groups_by_scene)<2:
        (a.output/'failure.json').write_text(json.dumps({'reason':'fewer than two scenes','requests':[m for m,_ in results]},indent=2)+'\n')
        raise ValueError('Not enough scenes')
    scenes=sorted(groups_by_scene);random.Random(a.seed).shuffle(scenes)
    tune_n=max(1,round(len(scenes)*0.2));tune=set(scenes[:tune_n])
    manifest={'dataset_revision':REV,'scope':'Bounded evenly spaced shard windows; scene-level train/tune partition; evaluation scene overlap unknown',
      'selection':{'seed':a.seed,'windows_per_shard':a.windows_per_shard,'window_mib':a.window_mib,'max_per_window':a.per_window,
      'excluded_scenes':sorted(excluded),'note':'Evenly spaced offsets and within-window samples, not uniform random samples of frames or scenes; no full shard SHA verification'},
      'groups':{'train':{'scenes':[],'samples':[]},'tune':{'scenes':[],'samples':[]}},'sources':[m for m,_ in results]}
    for scene in sorted(groups_by_scene):
        split='tune' if scene in tune else 'train';group=manifest['groups'][split];group['scenes'].append(scene)
        for member,(data,shard,header) in sorted(groups_by_scene[scene].items()):
            path=a.output/split/scene/Path(member).name;path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():raise ValueError(f'Duplicate output {path}')
            path.write_bytes(data)
            group['samples'].append({'path':str(path.resolve().relative_to(ROOT)),'source_member':member,
                'sha256':hashlib.sha256(data).hexdigest(),'shard':shard,'tar_header_offset':header})
    manifest['scene_disjoint']=True
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'scene_counts':{k:len(v['scenes']) for k,v in manifest['groups'].items()},
        'sample_counts':{k:len(v['samples']) for k,v in manifest['groups'].items()},
        'failed_requests':sum(bool(m.get('failed')) for m,_ in results)},indent=2))

if __name__ == '__main__':main()
