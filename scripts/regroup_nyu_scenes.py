#!/usr/bin/env python3
"""Create a fresh dataset split keeping potentially related scene names together."""
import argparse,hashlib,json,random,re,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def location_key(scene):
    return re.sub(r'(?<=\d)[a-z]+$', '',scene)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--exclude-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    if a.output.exists():p.error('output exists')
    try:a.output.resolve().relative_to(ROOT)
    except ValueError:p.error('output must be within project root')
    m=json.loads(a.manifest.read_text());old=json.loads(a.exclude_manifest.read_text())
    excluded={location_key(s) for g in old['groups'].values() for s in g['scenes']}
    scenes={};excluded_scenes=set()
    for group in m['groups'].values():
        for row in group['samples']:
            scene=Path(row['source_member']).parts[-2]
            if location_key(scene) in excluded:excluded_scenes.add(scene);continue
            scenes.setdefault(scene,[]).append(row)
    keys=sorted({location_key(scene) for scene in scenes})
    if len(keys)<2:raise ValueError('Not enough location groups')
    random.Random(a.seed).shuffle(keys);n=max(1,round(len(keys)*0.2));tune=set(keys[:n])
    a.output.mkdir(parents=True)
    out={'dataset_revision':m['dataset_revision'],'scope':'Conservative suffix-based location-group split; evaluation scene identity unknown',
         'source_manifest_sha256':sha(a.manifest),'exclude_manifest_sha256':sha(a.exclude_manifest),
         'selection':{'seed':a.seed,'location_group_rule':'strip trailing lower-case letters after final digit',
          'excluded_location_groups':sorted(excluded),'excluded_scenes':sorted(excluded_scenes),'tuning_group_fraction':0.2,
          'warning':'Location grouping is inferred from scene names, not verified metadata; near duplicates can remain.'},
         'groups':{'train':{'scenes':[],'samples':[]},'tune':{'scenes':[],'samples':[]}},'scene_disjoint':True}
    for scene in sorted(scenes):
        split='tune' if location_key(scene) in tune else 'train';group=out['groups'][split];group['scenes'].append(scene)
        for row in scenes[scene]:
            source=ROOT/row['path']
            if sha(source)!=row['sha256']:raise ValueError(f'Source hash mismatch: {source}')
            target=a.output/split/scene/Path(row['source_member']).name;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():raise ValueError(f'Duplicate output: {target}')
            shutil.copyfile(source,target)
            if sha(target)!=row['sha256']:raise ValueError(f'Copy hash mismatch: {target}')
            group['samples'].append({**row,'path':str(target.resolve().relative_to(ROOT)),'location_group':location_key(scene)})
    (a.output/'manifest.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'scene_counts':{k:len(v['scenes']) for k,v in out['groups'].items()},
       'sample_counts':{k:len(v['samples']) for k,v in out['groups'].items()},'excluded_scenes':sorted(excluded_scenes)},indent=2))

if __name__=='__main__':main()
