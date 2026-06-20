"""Tests for GP-PHS. Requires the [gp] extra (sklearn); skips cleanly without."""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from otwin.uq.gp_phs import GPPHS  # noqa: E402


def test_fit_predict_returns_mean_and_std() -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(60, 1))
    dXdt = -0.5 * X  # simple decay field
    gp = GPPHS(n_states=1, n_inputs=0)
    gp.fit(X, dXdt)
    mean, std = gp.predict(X, return_std=True)
    assert mean.shape == (60, 1)
    assert std.shape == (60, 1)
    assert np.all(std >= 0.0)
    # In-sample mean should track the target reasonably.
    assert np.sqrt(np.mean((mean - dXdt) ** 2)) < 0.1


def test_uncertainty_grows_away_from_data() -> None:
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, size=(40, 1))
    dXdt = np.sin(X)
    gp = GPPHS(n_states=1, n_inputs=0)
    gp.fit(X, dXdt)
    _, std_in = gp.predict(np.array([[0.0]]))
    _, std_out = gp.predict(np.array([[6.0]]))  # far from training support
    assert std_out[0, 0] > std_in[0, 0]


def test_predict_before_fit_raises() -> None:
    gp = GPPHS(n_states=1, n_inputs=0)
    with pytest.raises(RuntimeError):
        gp.predict(np.array([[0.0]]))
