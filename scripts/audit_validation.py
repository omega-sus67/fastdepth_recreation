#!/usr/bin/env python3
"""Record basic HDF5 schema, valid-depth distribution and corruption checks."""
import argparse,csv,hashlib,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import h5py,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data',type=Path,default=ROOT/'data/nyu_depth_v2/extracted/val/official')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args(); paths=sorted(a.data.glob('*.h5'))
if not paths:p.error('no samples')
if a.output.exists():p.error('output exists')
a.output.mkdir(parents=True)
schemas=Counter(); hashes={}; duplicate=[]; errors=[]; rows=[]
edges=np.linspace(0,12,121); hist=np.zeros(120,dtype=np.int64)
for path in paths:
 try:
  digest=hashlib.sha256(path.read_bytes()).hexdigest()
  if digest in hashes: duplicate.append([hashes[digest],path.name])
  hashes[digest]=path.name
  with h5py.File(path) as f:
   schema=tuple((k,str(f[k].shape),str(f[k].dtype)) for k in sorted(f.keys()))
   schemas[str(schema)]+=1
   depth=np.asarray(f['depth']); valid=depth[np.isfinite(depth)&(depth>0)]
   if not valid.size:raise ValueError('no positive finite depth')
   hist+=np.histogram(valid,bins=edges)[0]
   rows.append({'sample':path.name,'sha256':digest,'valid_fraction':valid.size/depth.size,'valid_min':float(valid.min()),'valid_max':float(valid.max()),'valid_mean':float(valid.mean())})
 except Exception as e: errors.append({'sample':path.name,'error':str(e)})
with (a.output/'samples.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['sample','sha256','valid_fraction','valid_min','valid_max','valid_mean']);w.writeheader();w.writerows(rows)
report={'files_found':len(paths),'files_read':len(rows),'schemas':dict(schemas),'exact_file_hash_duplicates':duplicate,'errors':errors,'depth_units':'unverified',
 'valid_fraction_min':min((x['valid_fraction'] for x in rows),default=None),'valid_fraction_mean':float(np.mean([x['valid_fraction'] for x in rows])) if rows else None,
 'positive_depth_min':min((x['valid_min'] for x in rows),default=None),'positive_depth_max':max((x['valid_max'] for x in rows),default=None),
 'histogram_bins':edges.tolist(),'histogram_counts':hist.tolist(),'histogram_note':'Native depth units; values beyond 12 omitted from displayed histogram.'}
(a.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
fig,ax=plt.subplots(figsize=(8,4));ax.stairs(hist,edges);ax.set(xlabel='Positive depth (native units; units unverified)',ylabel='Pixels',title='Local validation depth distribution');fig.tight_layout();fig.savefig(a.output/'depth_distribution.png',dpi=150);plt.close(fig)
print(json.dumps({k:v for k,v in report.items() if not k.startswith('histogram_')},indent=2))
