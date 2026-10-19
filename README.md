# PhaseBudgets

Reproducible artifact for **Sharp Phase Budgets for Moment-Calibrated Scalar Wiener–Hopf Factors**, Shivam Gupta (2026).

[Read the manuscript](Sharp_Phase_Budgets_Shivam_Gupta.pdf).

The problem is uncertainty propagation through a positive scalar factorization. A real log-kernel perturbation `h` is supported on `[-1,1]`, bounded by `epsilon`, and has zero polynomial moments through degree `2*r`. At normalized complex height `y` above the band centre, the exact separate budgets are

```text
phase = epsilon/pi * log(coth((r+1)*asinh(y)))
log-magnitude = 2*epsilon/pi * atan(exp(-(2*r+3)*asinh(y)))
```

The phase is unwrapped. The marginal bounds have different attaining functions; a corner of their bounding rectangle is not attainable. These are logarithmic calibration moments, not covariance moments. An even-spectrum constraint would make centre phase zero.

## Reproduce

Python 3.11 or newer is required; the recorded run used Python 3.12.14.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python run_experiments.py
python make_figures.py
python verify_symbolics.py
python -m unittest discover -s tests -v
```

To install the small exploratory command-line interface:

```bash
python -m pip install -e .
phasebudgets --height 0.1 --epsilon 0.02 --r 12 --tolerance 0.001
```

To build the manuscript with Tectonic:

```bash
cd manuscript
tectonic main.tex
```

## What is checked

- `src/phasebudgets.py`: stable scalar formulas, extremizers, calibration-order rule, and finite-cell LP proposals.
- `src/certifier.py`: separate exact moment repair, rational polynomial root isolation, and Arb antiderivative enclosures. LP convergence is not assumed for certificate soundness.
- `run_experiments.py`: 312 ball formula comparisons, 4,056 ball moment checks, 12 independent 70-digit quadratures, 18 off-centre/centre certificates, 10 refinement certificates, and 16 smooth examples.
- `verify_symbolics.py`: 14 exact symbolic polynomial divisions and 15 independent off-axis transform quadratures.
- `tests/test_research.py`: twelve test methods, including a separately assembled inexact-moment LP, higher-precision replay, malformed-proposal rejection and extreme-height formula checks.
- `results/`: raw proposals, rational root boxes, high-precision interval records, environment manifest, and test log.
- `manuscript/`: complete LaTeX sources, four figures, and two generated tables.

The general certifier currently supports constant envelopes, exact moments, rational locations and directions. The paper proves a broader weighted/inexact-moment dual formula; the implementation does not claim to handle all those cases. Smooth-case quadrature and the plotted joint sets are numerical illustrations. They are distinguished from validated continuum brackets in the paper.

## Interpretation

The reference method supplies a pointwise error budget conditional on an externally justified uncertainty class. It does not estimate an envelope from samples, certify a complete controller or PDE solution, or replace an existing factorization solver. The closed laws concern the band centre; off-centre calculations use optimization and certification.

The factorization representation, canonical annihilating signs, best-L1 duality, and general boundary instability are established mathematics. See the manuscript for the relation to earlier work and the specific sharp specialization studied here.

The experiment manifest contains timings only for reproducibility; no performance comparison is claimed. Numerical results are deterministic, apart from library/platform dependent LP choices and roundoff. Reproduction should compare certificate validity and numerical tolerances, not byte equality of timing fields.

## Citation

```bibtex
@unpublished{gupta2026phasebudgets,
  author = {Shivam Gupta},
  title = {Sharp Phase Budgets for Moment-Calibrated Scalar Wiener--Hopf Factors},
  year = {2026},
  note = {Research manuscript and reproducible software},
  url = {https://github.com/shi1720/phasebudgets}
}
```

Software is MIT licensed. The manuscript is a research manuscript, not a record of peer-reviewed acceptance.
