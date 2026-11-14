#!/usr/bin/env python3
"""Acquire original weights or an explicitly labeled, checksum-pinned community candidate."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def fetch(url, destination, timeout, expected=None):
    if destination.exists():
        actual = digest(destination)
        if expected and actual != expected:
            raise ValueError(f'Existing file has incorrect SHA-256: {destination}')
        return {'status': 'existing_hash_verified' if expected else 'existing_unverified',
                'path': str(destination.relative_to(ROOT)), 'bytes': destination.stat().st_size,
                'sha256': actual, 'source_url': url}
    temporary = destination.with_suffix(destination.suffix+'.part')
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, temporary.open('wb') as output:
            if 'text/html' in response.headers.get('Content-Type', ''):
                raise ValueError('Server returned HTML instead of a checkpoint')
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        actual = digest(temporary)
        if expected and actual != expected:
            raise ValueError('Downloaded checkpoint SHA-256 mismatch')
        if temporary.stat().st_size < 1024:
            raise ValueError('Downloaded file is too small to be a model checkpoint')
        temporary.replace(destination)
        return {'status': 'downloaded', 'path': str(destination.relative_to(ROOT)),
                'bytes': destination.stat().st_size, 'sha256': actual, 'source_url': url}
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--community-pruned', action='store_true', help='Acquire the community candidate, not authenticated original weights')
    parser.add_argument('--timeout', type=float, default=6)
    parser.add_argument('--report', type=Path, default=ROOT/'results/checkpoints/acquisition.json')
    args = parser.parse_args()
    config = json.loads((ROOT/'configs/checkpoints.json').read_text())
    report = {'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'attempts': []}
    if args.community_pruned:
        candidate = config['community_candidate']
        result = fetch(candidate['source_url'], ROOT/candidate['local_path'], args.timeout, candidate['sha256'])
        result.update(variant=candidate['variant'], source_status=candidate['source_status'])
        report['attempts'].append(result)
        complete = True
    else:
        def acquire(variant):
            attempts = []
            for scheme in ('https', 'http'):
                url = config['original_author_base_url'].replace('https:', scheme+':')+variant+'.pth.tar'
                try:
                    result = fetch(url, ROOT/'checkpoints/original'/f'{variant}.pth.tar', args.timeout)
                    result.update(variant=variant, source_status='original_author')
                    return result
                except (OSError, urllib.error.URLError, ValueError) as exc:
                    attempts.append({'source_url': url, 'error': str(exc)})
            return {'variant': variant, 'status': 'unavailable', 'attempts': attempts}
        with ThreadPoolExecutor(max_workers=4) as executor:
            report['attempts'] = list(executor.map(acquire, config['original_variants']))
        complete = all(item['status'] != 'unavailable' for item in report['attempts'])
    report['all_requested_downloads_available'] = complete
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    for result in report['attempts']:
        print(result['variant'], result['status'])
    print(f'Saved {args.report}')
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
