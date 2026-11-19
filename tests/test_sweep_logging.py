"""Exercise the sweep shell orchestration without requiring Jetson hardware."""
import csv
import os
from pathlib import Path
import subprocess

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_latency_sweep_jetson.sh'


def test_sweep_keeps_running_after_build_failure_and_groups_logs(tmp_path):
    text = SCRIPT.read_text()
    guard = 'if [ "$(uname -m)" != aarch64 ] || [ ! -f /etc/nv_tegra_release ]; then'
    assert guard in text
    # The test doubles below exercise orchestration, not the Jetson environment check.
    runner = tmp_path / 'run.sh'
    runner.write_text(text.replace(guard, 'if false; then', 1))
    for name in ('configs/experiment_2026-11-16_latency_sweep.json',
                 'data/nyu_depth_v2/jetson-tuning-445/manifest.json',
                 'data/nyu_depth_v2/jetson-evaluation-654/manifest.json',
                 'data/nyu_depth_v2/latency-tuning-references/manifest.json',
                 'data/nyu_depth_v2/latency-evaluation-references/manifest.json'):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    fakebin = tmp_path / 'bin'
    fakebin.mkdir()
    fake_bash = fakebin / 'bash'
    fake_bash.write_text('''#!/bin/sh
case "$3" in *width192_int8) echo simulated-build-error >&2; exit 1;; esac
mkdir -p "$3"
printf engine > "$3/model.engine"
''')
    fake_python = fakebin / 'python3'
    fake_python.write_text('''#!/usr/bin/python3
import pathlib, sys
if sys.argv[1] == '-c':
    if 'selected_student' in sys.argv[2]: print('width176')
else:
    args = sys.argv[1:]
    output = pathlib.Path(args[args.index('--output') + 1])
    if output.suffix == '.json':
        output.write_text('{"selected_student":"width176"}')
    else:
        output.mkdir(parents=True, exist_ok=True)
        (output / 'summary.json').write_text('{}')
''')
    for path in (fake_bash, fake_python):
        path.chmod(0o755)
    env = {**os.environ, 'PATH': f'{fakebin}:/usr/bin:/bin'}
    run = subprocess.run(['/bin/bash', str(runner), 'run'], cwd=tmp_path,
                         env=env, text=True, capture_output=True, timeout=30)
    assert run.returncode == 1, run.stdout + run.stderr
    out = tmp_path / 'run'
    with (out / 'status.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12
    failure = next(row for row in rows if row['model'] == 'width192' and row['precision'] == 'int8')
    assert failure['build_status'] == 'failed' and failure['tuning_status'] == 'not_run'
    assert 'simulated-build-error' in (out / 'logs/build/width192_int8.log').read_text()
    assert len(list((out / 'logs/build').glob('*.log'))) == 12
    assert len(list((out / 'logs/tuning').glob('*.log'))) == 11
    assert len(list((out / 'logs/evaluation').glob('*.log'))) == 6
    with (out / 'evaluation-status.csv').open(newline='') as f:
        assert len(list(csv.DictReader(f))) == 6
