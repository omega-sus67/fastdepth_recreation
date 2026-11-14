#!/usr/bin/env python3
"""Check available model shapes; random structural checks never imply pretrained accuracy."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from fastdepth_project.models import VARIANTS, build_random_model, load_converted, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--structural', action='store_true', help='Also check the three unpruned RANDOM architectures')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, default=ROOT/'results/checkpoints/model-checks.json')
    args = parser.parse_args()
    if not args.checkpoint and not args.structural:
        parser.error('Specify --checkpoint and/or --structural')
    results = []
    if args.structural:
        for variant in VARIANTS[:3]:
            torch.manual_seed(0)
            model = build_random_model(variant).to(args.device).eval()
            with torch.inference_mode():
                y = model(torch.zeros(1, 3, 224, 224, device=args.device))
            assert list(y.shape) == [1, 1, 224, 224] and torch.isfinite(y).all()
            results.append({'variant': variant, 'weights': 'random_structural_only',
                            'output_shape': list(y.shape), 'finite': True,
                            'parameter_count': sum(p.numel() for p in model.parameters())})
            del model
    if args.checkpoint:
        model, record = load_converted(args.checkpoint, args.device)
        rgb = np.load(ROOT/'third_party/fast-depth/deploy/data/rgb.npy')
        x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().unsqueeze(0).to(args.device)
        with torch.inference_mode():
            y = model(x).cpu().numpy()
        expected = np.load(ROOT/'third_party/fast-depth/deploy/data/pred.npy')
        assert list(y.shape) == [1, 1, 224, 224] and np.isfinite(y).all()
        results.append({'variant': record['variant'], 'weights': record['provenance']['source_status'],
                        'checkpoint_sha256': sha256(args.checkpoint),
                        'parameter_count': sum(p.numel() for p in model.parameters()),
                        'output_shape': list(y.shape), 'finite': True,
                        'upstream_example_comparison': {
                            'max_abs_error': float(np.abs(y-expected).max()),
                            'mean_abs_error': float(np.abs(y-expected).mean()),
                            'allclose_rtol_1e-3_atol_1e-3': bool(np.allclose(y, expected, rtol=1e-3, atol=1e-3)),
                            'notes': 'Single vendored TVM-example comparison; neither source authentication nor dataset accuracy.'}})
    report = {'device': args.device, 'input_shape': [1,3,224,224], 'accuracy_verified': False, 'models': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
