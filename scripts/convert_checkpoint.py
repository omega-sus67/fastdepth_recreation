#!/usr/bin/env python3
"""Convert a provenance-reviewed legacy FastDepth checkpoint to explicit config/weights."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import torch
from fastdepth_project.models import VARIANTS, from_spec, load_converted, load_legacy, module_spec, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--sha256', required=True, help='Expected SHA-256 after source/provenance review')
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--source-url', required=True)
    parser.add_argument('--source-status', choices=['original_author', 'community_mirror_unverified'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new path to preserve the previous conversion')
    legacy, checkpoint = load_legacy(args.input, args.sha256)
    config = module_spec(legacy)
    model = from_spec(config)
    model.load_state_dict(legacy.state_dict(), strict=True)
    model.eval().to(args.device)
    generator = torch.Generator().manual_seed(0)
    x = torch.rand((1, 3, 224, 224), generator=generator).to(args.device)
    with torch.inference_mode():
        output = model(x)
    if list(output.shape) != [1, 1, 224, 224] or not torch.isfinite(output).all():
        raise ValueError('Converted forward output is invalid')
    provenance = {'source_url': args.source_url, 'source_status': args.source_status,
                  'sha256': args.sha256, 'converted_at_utc': datetime.now(timezone.utc).isoformat(),
                  'original_class': type(legacy).__name__, 'torch_version': str(torch.__version__),
                  'upstream_revision': 'e68492011609c9bfb7de6d402da5d1d201d95bd9'}
    record = {'schema_version': 1, 'variant': args.variant, 'model_config': config,
              'state_dict': {key: value.detach().cpu() for key, value in model.state_dict().items()},
              'provenance': provenance}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(record, args.output)
    args.output.with_suffix('.architecture.json').write_text(json.dumps(config, indent=2)+'\n')
    restored, _ = load_converted(args.output, args.device)
    with torch.inference_mode():
        restored_output = restored(x)
    torch.testing.assert_close(restored_output, output, rtol=0, atol=0)
    legacy_parity = {'status': 'not_executable', 'reason': None}
    try:
        legacy.eval().to(args.device)
        with torch.inference_mode():
            old_output = legacy(x)
        torch.testing.assert_close(old_output, output, rtol=1e-5, atol=1e-6)
        legacy_parity = {'status': 'passed', 'max_abs_error': float((old_output-output).abs().max())}
    except (AttributeError, RuntimeError, TypeError, AssertionError) as exc:
        legacy_parity['reason'] = str(exc)
    report = {'provenance': provenance, 'variant': args.variant, 'device': args.device,
              'input_shape': list(x.shape), 'output_shape': list(output.shape),
              'finite': bool(torch.isfinite(output).all()),
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'converted_sha256': sha256(args.output), 'reload_bit_exact': True,
              'legacy_parity': legacy_parity,
              'accuracy_verified': False, 'original_author_identity_verified': args.source_status == 'original_author',
              'notes': 'Synthetic forward/conversion checks are not pretrained accuracy or original-paper reproduction.'}
    if isinstance(checkpoint, dict):
        report['legacy_metadata_keys'] = list(checkpoint)
        result = checkpoint.get('best_result')
        if result is not None and hasattr(result, '__dict__'):
            values = {key: value for key, value in vars(result).items() if isinstance(value, (int, float))}
            report['legacy_claimed_metrics_not_remeasured'] = {
                key: value if math.isfinite(value) else None for key, value in values.items()}
            report['legacy_nonfinite_metric_fields'] = [key for key, value in values.items() if not math.isfinite(value)]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('variant','output_shape','parameter_count','reload_bit_exact','legacy_parity')}, indent=2))


if __name__ == '__main__':
    main()
