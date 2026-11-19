import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('selection', Path(__file__).parents[1] / 'scripts/select_latency_candidate.py')
selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)


def result(rmse, delta, rounds, manifest='same'):
    return {'pooled': {'rmse': rmse, 'delta1': delta}, 'round_medians_ms': rounds,
            'manifest_sha256': manifest, 'samples': 445}


def test_rejects_inaccurate_faster_student():
    baseline = result(1.0, 0.9, [10, 10.1, 10])
    candidates = {'fast': result(1.2, 0.9, [5, 5.1, 5]),
                  'valid': result(1.0, 0.895, [8, 8.1, 8])}
    assert selection.choose(baseline, candidates, .1, .01)['selected_student'] == 'valid'


def test_close_latencies_choose_lower_rmse():
    baseline = result(1, .9, [10, 10, 10])
    candidates = {'fast': result(.99, .9, [7.9, 8.0, 8.1]),
                  'accurate': result(.95, .9, [8.1, 8.1, 8.1])}
    assert selection.choose(baseline, candidates, .1, .01)['selected_student'] == 'accurate'


def test_mismatched_data_fails():
    with pytest.raises(ValueError, match='Mismatched'):
        selection.choose(result(1, .9, [10, 10]), {'x': result(1, .9, [8, 8], 'other')}, .1, .01)


def test_no_eligible_candidate():
    assert selection.choose(result(1, .9, [10, 10]), {'x': result(2, .9, [8, 8])}, .1, .01)['selected_student'] is None
