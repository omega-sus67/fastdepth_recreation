#!/usr/bin/env python3
"""Select a tuning-eligible candidate using recorded FP16-enabled timing rounds."""
import argparse
import json
import statistics
from pathlib import Path


def choose(baseline, candidates, max_rmse_increase, max_delta_drop):
    """Return candidate and auditable gate decisions; no final-evaluation data."""
    if len(baseline['round_medians_ms']) < 2:
        raise ValueError('At least two timing rounds required')
    base_rmse = baseline['pooled']['rmse']; base_delta = baseline['pooled']['delta1']
    if base_rmse <= 0: raise ValueError('Baseline RMSE must be positive')
    records = []
    for name, result in candidates.items():
        if result['manifest_sha256'] != baseline['manifest_sha256'] or result['samples'] != baseline['samples']:
            raise ValueError(f'Mismatched tuning data: {name}')
        rounds = result['round_medians_ms']
        if len(rounds) < 2: raise ValueError(f'Insufficient timing rounds: {name}')
        rmse, delta = result['pooled']['rmse'], result['pooled']['delta1']
        records.append({'name': name, 'pooled_rmse': rmse, 'delta1': delta,
                        'eligible': rmse <= base_rmse * (1 + max_rmse_increase) and delta >= base_delta - max_delta_drop,
                        'median_of_round_medians_ms': statistics.median(rounds),
                        'observed_round_range_ms': max(rounds) - min(rounds)})
    eligible = sorted((x for x in records if x['eligible']), key=lambda x: x['median_of_round_medians_ms'])
    if not eligible: return {'selected_student': None, 'candidates': records, 'reason': 'No candidate passed both accuracy gates'}
    best = eligible[0]
    near = [x for x in eligible if x['median_of_round_medians_ms'] - best['median_of_round_medians_ms'] <= max(x['observed_round_range_ms'], best['observed_round_range_ms'])]
    selected = min(near, key=lambda x: (x['pooled_rmse'], x['name']))
    baseline_median = statistics.median(baseline['round_medians_ms'])
    variation = max(max(baseline['round_medians_ms']) - min(baseline['round_medians_ms']), selected['observed_round_range_ms'])
    return {'selected_student': selected['name'], 'candidates': records, 'baseline_median_of_round_medians_ms': baseline_median,
            'observed_latency_difference_ms': baseline_median - selected['median_of_round_medians_ms'],
            'difference_exceeds_observed_round_range': baseline_median - selected['median_of_round_medians_ms'] > variation,
            'reason': 'Fastest eligible candidate; within observed round-range ties choose lower pooled RMSE',
            'limits': 'Simple variability heuristic, not a statistical significance test; timings may drift with order, temperature or power.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True); p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    if a.output.exists(): p.error('output exists')
    cfg = json.loads(a.config.read_text())['selection']
    baseline = json.loads((a.runs / 'baseline_fp16' / 'summary.json').read_text())
    candidates = {}; missing = []
    for name in ('width192','width176','width160'):
        path = a.runs / f'{name}_fp16' / 'summary.json'
        if path.exists(): candidates[name] = json.loads(path.read_text())
        else: missing.append(name)
    result = choose(baseline, candidates, cfg['maximum_relative_rmse_increase'], cfg['maximum_absolute_delta1_drop'])
    result.update({'selection_scope': 'FP16-enabled engines; already-inspected tuning set only',
                   'accuracy_gates': cfg, 'missing_candidates': missing,
                   'complete_candidate_set': not missing,
                   'baseline_pooled': baseline['pooled'], 'baseline_samples': baseline['samples'],
                   'tuning_manifest_sha256': baseline['manifest_sha256']})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
