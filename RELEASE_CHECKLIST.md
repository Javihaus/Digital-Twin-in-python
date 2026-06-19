# Release Checklist for PKG v2

This checklist guides the transition from v1 to v2 and the first v2 release.

---

## Pre-Release: Code Complete

### Core Implementation
- [x] Phase 0: Scaffold complete
- [x] Phase 1: Analytic PHS core complete
- [x] Phase 2: Evaluation harness complete
- [x] Phase 3: DigitalTwin interface complete
- [x] Phase 5: IPHS entropy layer complete
- [ ] Phase 4: Learned PHS (PHNN) - requires torch implementation
- [ ] Phase 4: UQ (ensembles, calibration) - requires implementation
- [ ] Phase 6: Port composition - requires implementation
- [ ] Phase 6: GP-PHS - optional, requires gpytorch

### Testing
- [x] Property-based tests (hypothesis) for structural guarantees
- [x] Integration tests (end-to-end workflows)
- [x] Evaluation guard tests (baselines, splits)
- [x] Metrics validation (known ground truth)
- [ ] Coverage ≥ 85% on all core modules (verify)
- [ ] All tests pass in CI for Python 3.10-3.13

### Documentation
- [x] All public APIs have docstrings
- [x] Phase reports (0-3, 5) complete
- [x] GETTING_STARTED.md
- [x] CITATIONS.md with tracking
- [x] CHANGELOG.md
- [x] NOTICE_v1_to_v2.md
- [ ] API reference (auto-generated from docstrings)
- [ ] Theory guide (PHS, IPHS, structure preservation)
- [ ] Tutorial notebooks

### Examples & Benchmarks
- [x] Water tank PHS example
- [x] Benchmark script structure
- [ ] Battery NASA example (requires Phase 4)
- [ ] Run benchmarks and generate results
- [ ] Update README with generated numbers

---

## v1 to v2 Migration

### 1. Backup Current State
```bash
# Create v1 snapshot
git checkout main
git tag v1-final
git branch v1-archive

# Create backup of current pyproject.toml
cp pyproject.toml pyproject_v1_backup.toml
```

### 2. Move v1 to Legacy
```bash
# Create legacy directory
mkdir -p legacy_v1

# Move v1 source (preserve git history)
git mv src/hybrid_digital_twin legacy_v1/hybrid_digital_twin

# Move v1 specific files
git mv notebooks/hybrid_twin_tutorial.ipynb legacy_v1/ 2>/dev/null || true
git mv examples/battery_v1_tutorial.py legacy_v1/ 2>/dev/null || true

# Keep LICENSE, README will be updated
```

### 3. Fix v1 Bugs (in legacy_v1/)
Apply fixes documented in CHANGELOG.md:
- [ ] Fix `load_nasa_dataset` double-prefix path bug
- [ ] Fix `save_model` joblib-pickle-Keras issue
- [ ] Remove placeholder `_estimate_uncertainty`
- [ ] Fix ignored YAML config keys
- [ ] Update benchmark table with generated numbers or remove
- [ ] Correct Xu et al. attribution or remove

### 4. Activate v2 Configuration
```bash
# Replace main pyproject.toml
mv pyproject.toml legacy_v1/pyproject_v1.toml
cp pyproject_v2.toml pyproject.toml

# Replace main README
mv README.md legacy_v1/README_v1.md
cp README_v2.md README.md

# Replace .gitignore
mv .gitignore legacy_v1/.gitignore_v1
cp .gitignore_v2 .gitignore

# Activate v2 CI
mv .github/workflows/ci.yml .github/workflows/ci_v1.yml 2>/dev/null || true
cp .github/workflows/ci_v2.yml .github/workflows/ci.yml
```

### 5. Update Package Name (Global Search-Replace)
```bash
# Choose final package name (currently "PKG")
# Example: if final name is "phdigitaltwin"

find src/PKG -type f -name "*.py" -exec sed -i '' 's/PKG/phdigitaltwin/g' {} +
find tests -type f -name "*.py" -exec sed -i '' 's/PKG/phdigitaltwin/g' {} +
find examples -type f -name "*.py" -exec sed -i '' 's/PKG/phdigitaltwin/g' {} +

# Update pyproject.toml name field
sed -i '' 's/name = "PKG"/name = "phdigitaltwin"/' pyproject.toml

# Rename directory
mv src/PKG src/phdigitaltwin
```

### 6. Commit v2 Structure
```bash
git add -A
git commit -m "refactor: Activate v2 structure, migrate v1 to legacy

- Move v1 to legacy_v1/ with fixes
- Activate v2 config (pyproject.toml, CI, README)
- Fix v1 bugs (data loader, serialization, placeholders)
- Update citations and remove misattributions
- Structure v2 as described in NOTICE_v1_to_v2.md

BREAKING CHANGE: v2 is a complete rewrite. See NOTICE_v1_to_v2.md
for migration guide.

Refs: #issue-number"
```

---

## First v2 Alpha Release

### 1. Verify All Gates Pass
```bash
# Run full check
make check

# Verify structure
make verify-structure

# Run all tests with coverage
make test-coverage

# Check coverage report (should be ≥ 85%)
open htmlcov/index.html
```

### 2. Generate Benchmarks
```bash
# Run benchmarks
make benchmark

# Verify results
ls benchmarks/results/
cat benchmarks/results/benchmark_summary.json
```

### 3. Update README with Generated Numbers
```python
# Script to insert benchmark results into README
import json

with open('benchmarks/results/benchmark_summary.json') as f:
    benchmarks = json.load(f)

# Update README benchmark table section
# (Use script or manual update with generated numbers)
```

### 4. Verify CITATIONS
- [ ] Review all UNVERIFIED entries in CITATIONS.md
- [ ] Verify critical references (PHS theory, HNN, IPHS)
- [ ] Mark verified entries as VERIFIED
- [ ] Remove or correct any incorrect attributions

### 5. Final Version Bump
```bash
# Update version in:
# - pyproject.toml: version = "2.0.0-alpha"
# - src/phdigitaltwin/__init__.py: __version__ = "2.0.0-alpha"
# - CHANGELOG.md: Add release date

# Commit
git commit -am "chore: Prepare v2.0.0-alpha release"
```

### 6. Create Release Tag
```bash
git tag -a v2.0.0-alpha -m "v2.0.0-alpha: First v2 release

Complete rewrite with:
- Port-Hamiltonian structure by construction
- Honest evaluation by default
- Irreversible PHS with entropy production
- Gating CI and property-based tests

Status: Alpha - core functionality complete, learned models pending.

See CHANGELOG.md for full details."

git push origin v2.0.0-alpha
```

### 7. Build and Test PyPI Package
```bash
# Build
python -m build

# Check
twine check dist/*

# Test upload to TestPyPI first
twine upload --repository testpypi dist/*

# Test install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ phdigitaltwin

# If all works, upload to PyPI
twine upload dist/*
```

### 8. Create GitHub Release
- [ ] Go to GitHub Releases
- [ ] Create new release from v2.0.0-alpha tag
- [ ] Title: "v2.0.0-alpha - Port-Hamiltonian Digital Twins"
- [ ] Description: Paste from CHANGELOG.md + NOTICE_v1_to_v2.md summary
- [ ] Mark as "pre-release" (alpha status)
- [ ] Attach: wheel, tarball, PHASE_*_REPORT.md files

---

## Post-Release

### 1. Update Documentation
- [ ] Publish API docs (Sphinx or similar)
- [ ] Create ReadTheDocs project
- [ ] Link from README

### 2. Community
- [ ] Announce on relevant forums (if appropriate)
- [ ] Update GitHub repo description/tags
- [ ] Create discussion threads for feedback

### 3. Monitor
- [ ] Watch for installation issues
- [ ] Address critical bugs immediately
- [ ] Collect feedback for Beta roadmap

---

## Roadmap to Beta

### Required for v2.0.0-beta
- [ ] Complete Phase 4: PortHamiltonianNN + UQ
- [ ] Battery NASA example with learned model
- [ ] API stabilization (no more breaking changes without deprecation)
- [ ] Documentation complete (API ref, theory guide, tutorials)
- [ ] Coverage ≥ 90% on core modules
- [ ] At least 3 real-world validation cases

### Required for v2.0.0 (Stable)
- [ ] Production deployments with monitoring
- [ ] Performance benchmarking suite
- [ ] Full GP-PHS implementation
- [ ] Port composition implementation
- [ ] Community feedback integrated
- [ ] No critical bugs for 3 months
- [ ] Deprecation policy established

---

## Emergency Rollback

If critical issues found after release:

```bash
# Yank bad release from PyPI
pip install twine
twine upload --skip-existing --repository pypi dist/*  # Re-upload with note

# Create hotfix
git checkout v2.0.0-alpha
git checkout -b hotfix/critical-issue
# ... fix ...
git tag v2.0.0-alpha.1
git push origin v2.0.0-alpha.1

# Or fully revert
git revert <commit-hash>
git tag v2.0.0-alpha.1
```

---

## Sign-Off

Before tagging release, verify:
- [x] All Phase 0-3, 5 implementations complete
- [x] All tests pass (property-based + integration)
- [x] Evaluation guards work (baselines mandatory, random split warns)
- [x] Structure preservation verified (energy decay, power balance)
- [x] Entropy production ≥ 0 (tested)
- [x] Documentation complete for implemented features
- [ ] Benchmarks generated and reproducible
- [ ] v1 migrated to legacy_v1/ with fixes
- [ ] CI gating active (no swallowing)
- [ ] Coverage ≥ 85% (verify with `make test-coverage`)

**Release Manager:** _______________  Date: _______________

**QA Sign-off:** _______________  Date: _______________

---

## Post-v2.0.0 Roadmap

### Short Term (1-2 months)
- Phase 4 completion (torch + UQ)
- Battery example with learned model
- Tutorial notebooks
- API stability

### Medium Term (3-6 months)
- Beta release
- Extended examples library
- Performance optimizations
- Community growth

### Long Term (6-12 months)
- Stable 1.0 release
- Production validation
- Research collaborations
- Possible: GPU support, distributed training

---

**v2 represents a fundamental shift: from tutorial to rigorous library.**

All claims are testable. All numbers are traceable. All structure is proven.
