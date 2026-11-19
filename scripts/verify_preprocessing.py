#!/usr/bin/env python3
"""Compare project preprocessing to functions extracted from SciPy 1.1.0 source."""
import argparse,ast,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image
from fastdepth_project.data import load_h5,preprocess
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,default=ROOT/'results/data/source_and_preprocessing/scipy-pilutil-1.1.0.py')
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():p.error('output exists')
source=a.source.read_bytes()
expected_sha256='2115336ef6812bd90ec9d8a4481c6769c633476f658f298a4a1ccd418e8ffe47'
if hashlib.sha256(source).hexdigest()!=expected_sha256:
 raise ValueError('Historical SciPy source hash mismatch; refusing to execute modified code')
# Copy only the four upstream function definitions, omitting obsolete decorators;
# preserve their bodies. Modern Pillow removed the legacy type-check helper.
tree=ast.parse(source.decode());names={'bytescale','toimage','fromimage','imresize'}
funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
assert {f.name for f in funcs}==names
for f in funcs:f.decorator_list=[]
class NumpyCompatibility(ast.NodeTransformer):
 def visit_Attribute(self,node):
  self.generic_visit(node)
  if node.attr=='tostring':node.attr='tobytes'
  return node
funcs=[NumpyCompatibility().visit(f) for f in funcs]
module=ast.fix_missing_locations(ast.Module(body=funcs,type_ignores=[]))
Image.isImageType=lambda im:isinstance(im,Image.Image)
env={'numpy':np,'Image':Image,'_errstr':'Mode incompatible'}
for name in ('amin','amax','ravel','asarray','arange','ones','newaxis','transpose','iscomplexobj','uint8','issubdtype','array'):
 env[name]=getattr(np,name)
exec(compile(module,str(a.source),'exec'),env)
def legacy(rgb,depth):
 def transform(x):
  x=env['imresize'](x,250.0/480,'nearest',None if x.ndim==3 else 'F')
  y=round((x.shape[0]-228)/2);z=round((x.shape[1]-304)/2)
  return env['imresize'](x[y:y+228,z:z+304],(224,224),'nearest',None if x.ndim==3 else 'F')
 return transform(rgb).astype(np.float64)/255,transform(depth)
rng=np.random.default_rng(42);cases=[]
inputs=[('synthetic-pattern',rng.integers(0,256,(480,640,3),dtype=np.uint8),rng.uniform(.1,10,(480,640)).astype(np.float32))]
for path in sorted((ROOT/'data/nyu_depth_v2/extracted/val/official').glob('*.h5'))[:5]:
 rgb,depth=load_h5(path);inputs.append((path.name,rgb,depth))
for name,rgb,depth in inputs:
 ref_x,ref_y=legacy(rgb,depth);x,y=preprocess(rgb,depth);new_x=x.permute(1,2,0).numpy();new_y=y[0].numpy()
 cases.append({'case':name,'rgb_uint8_reconstructed_equal':bool(np.array_equal((new_x*255).round().astype(np.uint8),(ref_x*255).round().astype(np.uint8))),
  'rgb_float_max_abs_diff':float(np.abs(ref_x-new_x).max()),
  'rgb_after_legacy_float32_cast_exact_equal':bool(np.array_equal(ref_x.astype(np.float32),new_x)),
  'depth_exact_equal':bool(np.array_equal(ref_y,new_y)),
  'depth_max_abs_diff':float(np.abs(ref_y-new_y).max())})
report={'reference_source':'https://github.com/scipy/scipy/blob/v1.1.0/scipy/misc/pilutil.py','source_sha256':hashlib.sha256(source).hexdigest(),
 'method':'SciPy 1.1.0 functions; decorators removed, ndarray.tostring replaced by equivalent tobytes and Pillow type-check shim; current NumPy/Pillow',
 'limitation':'Not a historical SciPy/Pillow binary environment; five real files plus one synthetic case; comparison uses float32 cast matching upstream ToTensor.',
 'cases':cases,'all_pixel_coordinates_match':all(x['rgb_uint8_reconstructed_equal'] and x['depth_exact_equal'] for x in cases),
 'all_final_tensors_exact':all(x['rgb_after_legacy_float32_cast_exact_equal'] and x['depth_exact_equal'] for x in cases)}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if not report['all_final_tensors_exact']:raise SystemExit('Preprocessing discrepancy')
