#!/usr/bin/env python3
"""Extract downloaded validation HDF5 files under data/nyu_depth_v2/extracted."""
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parents[1]
archives = sorted((root / 'data/nyu_depth_v2/raw').glob('val-*.tar'))
if len(archives) != 2:
    raise SystemExit('Expected both validation archives; run scripts/download_validation.sh first.')
dest = root / 'data/nyu_depth_v2/extracted'
dest.mkdir(parents=True, exist_ok=True)
count = 0
for archive in archives:
    with tarfile.open(archive) as tar:
        for member in tar:
            path = Path(member.name)
            if not member.isfile() or path.is_absolute() or '..' in path.parts or path.suffix != '.h5' or path.parts[:2] != ('val', 'official'):
                raise ValueError(f'Unexpected archive member: {member.name}')
            target = dest / path
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open('wb') as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
            count += 1
print(f'Extracted {count} samples to {dest / "val"}')
