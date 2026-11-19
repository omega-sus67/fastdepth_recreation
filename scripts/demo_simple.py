#!/usr/bin/env python3
"""Run the selected baseline or student on one 640x480 RGB image."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--model', choices=('baseline', 'student'), required=True)
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--device', help='Optional PyTorch device, e.g. cpu or cuda')
    parser.add_argument('--checkpoint', type=Path, help='Override the old fixed selection with the board-selected checkpoint')
    args = parser.parse_args()
    selection = json.loads((ROOT / 'configs/selection_2026-11-16_simple.json').read_text())
    checkpoint = args.checkpoint if args.checkpoint else ROOT / selection[args.model]
    if not checkpoint.is_file():
        parser.error(f'selected {args.model} checkpoint not found: {checkpoint}')
    command = [sys.executable, str(ROOT / 'scripts/infer_image.py'), str(args.image),
               '--checkpoint', str(checkpoint), '--output', str(args.output)]
    if args.device:
        command.extend(['--device', args.device])
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
