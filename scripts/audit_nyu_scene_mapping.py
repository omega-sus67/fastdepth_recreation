#!/usr/bin/env python3
"""Infer scene identities from numeric filenames and verify against matched prefix."""
import argparse,hashlib,json,re,sys,urllib.request
from pathlib import Path
import h5py,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from audit_official_nyu_identity import RangeFile,URL

def group_key(scene):return re.sub(r'(?<=\d)[a-z]+$','',scene)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
 p.add_argument('--prefix-check',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error('output exists')
 m=json.loads(a.manifest.read_text());evidence=json.loads(a.prefix_check.read_text());matches=evidence['matches']
 if len(matches)<1:raise ValueError('No exact matches supporting filename mapping')
 for match in matches:
  expected=f"{match['zero_based_official_index']+1:05d}.h5"
  if match['local_evaluation_files']!=[expected]:raise ValueError('Exact match contradicts index hypothesis')
 paths=sorted((ROOT/'data/nyu_depth_v2/extracted/val/official').glob('*.h5'))
 if not paths or any(not p.stem.isdigit() or not 1<=int(p.stem)<=1449 for p in paths):raise ValueError('Unsupported evaluation filenames')
 with urllib.request.urlopen(urllib.request.Request(URL,method='HEAD'),timeout=30) as response:size=int(response.headers['Content-Length'])
 remote=RangeFile(URL,size);mapping=[]
 with h5py.File(remote,'r') as f:
  refs=f['scenes']
  if refs.shape!=(1,1449):raise ValueError('Unexpected scenes shape')
  for i,path in enumerate(paths,1):
   index=int(path.stem)-1;ref=refs[0,index];scene=''.join(chr(int(c)) for c in np.asarray(f[ref]).ravel())
   mapping.append({'local_evaluation_file':path.name,'inferred_official_zero_based_index':index,'inferred_scene':scene})
   if i%100==0:print(f'Mapped {i}/{len(paths)} inferred evaluation scenes',flush=True)
 train={group_key(s):[] for s in m['groups']['train']['scenes']};tune={group_key(s):[] for s in m['groups']['tune']['scenes']}
 for label,groups in [('train',train),('tune',tune)]:
  for s in m['groups'][label]['scenes']:groups[group_key(s)].append(s)
 overlaps={'train':[],'tune':[]}
 for row in mapping:
  key=group_key(row['inferred_scene'])
  for label,groups in [('train',train),('tune',tune)]:
   if key in groups:overlaps[label].append({**row,'matching_sampled_scenes':groups[key]})
 report={'scope':'Inferred filename-to-official-index scene mapping; only the prefix matches were independently verified',
  'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),'prefix_evidence_sha256':hashlib.sha256(a.prefix_check.read_bytes()).hexdigest(),
  'evidence_exact_matches':len(matches),'mapping_rule':'local five-digit filename number minus one equals original NYU labeled image index',
  'group_rule':'strip trailing lowercase suffix after final number; heuristic, not verified location identity',
  'evaluation_files':len(paths),'inferred_evaluation_scene_count':len({r['inferred_scene'] for r in mapping}),
  'potential_overlap':overlaps,'mapping':mapping,'remote_requests':remote.requests,'remote_bytes_fetched':remote.bytes_fetched,
  'limitations':'Unverified indices are inferred from filenames, not exact RGB matched; source names are not proof of independent or same physical locations. Community checkpoint pretraining remains unknown.'}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k!='mapping'},indent=2))

if __name__=='__main__':main()
