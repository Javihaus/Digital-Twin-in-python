# legacy_v1 — original Hybrid Digital Twin tutorial

This is the **original** tutorial code (the "Hybrid Digital Twin for Li-ion
battery modeling") that the repository started as. It is kept here for
provenance and to honour the project's history, **not** for active use.

Use the v2 library (`otwin/`) instead. See `../STATUS.md` and `../README.md`.

## Known issues in this legacy code (documented, not fixed)

This code is preserved as-is; the following were identified during review and are
**not** corrected here (they motivated the v2 rewrite):

- The benchmark table in the original README was not reproducible from the repo
  (no seeded script, no declared split).
- The reported headline used a random split (interpolation), not a temporal split
  (forecasting); under temporal split the picture is very different.
- The degradation formula `f_d = k·T·i/t` was attributed to Xu et al. (2016); it
  is an ad-hoc simplification, not the Xu semi-empirical model.
- Hybrid `save_model` pickled a Keras model unreliably; the canonical NASA loader
  double-prefixed the data directory; config keys did not reach the physics model;
  uncertainty was a placeholder constant.
- CI passed with `continue-on-error`, so a green check did not mean tests passed.

These tests are excluded from the default `pytest` run and the package build.
They require the legacy dependencies (TensorFlow/Keras, joblib) which the v2 core
does not.
