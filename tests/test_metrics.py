import numpy as np
import pytest
from fastdepth_project.metrics import DepthMetrics

def test_known_values_and_invalid_targets():
    m = DepthMetrics()
    m.update([1, 3, 100], [1, 2, 0])
    r = m.result()
    assert r['valid_pixels'] == 2
    assert r['rmse'] == pytest.approx(2**-0.5)
    assert r['mae'] == pytest.approx(.5)
    assert r['rel'] == pytest.approx(.25)
    assert r['delta1'] == pytest.approx(.5)
    assert r['r2'] == pytest.approx(-1)

def test_invalid_predictions_are_not_dropped():
    m = DepthMetrics()
    m.update([0, -1], [1, 2])
    assert m.result()['nonpositive_predictions'] == 2
    assert m.result()['valid_pixels'] == 2
    assert m.result()['delta1'] == 0
    with pytest.raises(ValueError, match='nonfinite'):
        m.update([np.nan], [1])

def test_empty_constant_and_shape():
    m = DepthMetrics()
    m.update([1], [0])
    with pytest.raises(ValueError, match='no valid'):
        m.result()
    m.update([1, 1], [2, 2])
    assert m.result()['r2'] is None
    with pytest.raises(ValueError, match='shapes'):
        m.update([1], [1, 2])
