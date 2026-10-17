"""Moment-calibrated uncertainty for positive scalar Wiener--Hopf symbols.

All phases are unwrapped imaginary parts of the normalized analytic logarithm.
The closed formulas apply at the centre of a finite uncertain band only.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import linprog
from numpy.polynomial import Legendre, Polynomial


def _log_coth(s: float) -> float:
    if s < 0.5:
        return -math.log(math.tanh(s))
    q = math.exp(-2*s)
    return math.log1p(q)-math.log1p(-q)


def phase_budget(r: int, y: float, epsilon: float = 1.0) -> float:
    """Sharp phase budget with moments of degrees 0,...,2r fixed."""
    if r < 0 or int(r) != r or y <= 0 or epsilon < 0:
        raise ValueError("r integer >=0, y>0, epsilon>=0 required")
    return epsilon / math.pi * _log_coth((r+1)*math.asinh(y))


def gain_budget(r: int, y: float, epsilon: float = 1.0) -> float:
    """Sharp log-magnitude budget under the same moment constraints."""
    if r < 0 or int(r) != r or y <= 0 or epsilon < 0:
        raise ValueError("r integer >=0, y>0, epsilon>=0 required")
    return 2 * epsilon / math.pi * math.atan(math.exp(-(2*r+3)*math.asinh(y)))


def required_order(y: float, epsilon: float, tau: float) -> int:
    """Smallest r with phase_budget(r,y,epsilon) <= tau; exact-arithmetic rule.

    Near integer thresholds, verify with ball arithmetic rather than float ceil.
    """
    if y <= 0 or epsilon < 0 or tau <= 0:
        raise ValueError("y,tau>0 and epsilon>=0 required")
    if epsilon == 0:
        return 0
    target = math.pi * tau / epsilon
    threshold = (0.5*_log_coth(target/2)) / math.asinh(y)
    return max(0, math.ceil(threshold) - 1)


def noisy_first_moment_budget(y: float, delta: float, epsilon: float = 1) -> float:
    """Mean and second moment fixed, |integral t*h| <= delta, |h|<=epsilon."""
    if y <= 0 or epsilon <= 0 or delta < 0:
        raise ValueError("y,epsilon>0, delta>=0 required")
    a2 = (1 + min(delta/epsilon, 1.0))/2
    return epsilon/(2*math.pi)*math.log((y*y+a2)**2/(y*y*(1+y*y)))


def extremizer(r: int, kind: str = 'phase'):
    """Exact mathematical switch nodes, float representation for exploration.

    On each returned cell, h=sign*epsilon. Phase witness gives negative phase.
    """
    if kind not in ('phase', 'gain'):
        raise ValueError('phase or gain required')
    n = 2*r + (1 if kind == 'phase' else 2)
    edges = np.r_[-1., np.sort(np.cos(np.arange(1,n+1)*np.pi/(n+1))), 1.]
    mid = (edges[:-1]+edges[1:])/2
    u = np.sin((n+1)*np.arccos(mid))/np.sqrt(1-mid*mid)
    sign = np.sign(((-1)**(r if kind == 'phase' else r+1))*u)
    return edges, sign


def cell_integrals(edges, x, y):
    """Integral of y/denom and -(t-x)/denom, without 1/(2*pi)."""
    u = np.asarray(edges)-x
    return np.diff(np.arctan(u/y)), -0.5*np.diff(np.log(u*u+y*y))


def optimize_cells(m, x, y, direction=(0.,1.), cells=512):
    """Finite feasible search; returns exact-cell LP and a continuum dual candidate.

    LP results alone are NOT certificates. certifier.py repairs moments in ball
    arithmetic and integrates the continuum dual using rational root isolation.
    """
    edges = np.linspace(-1,1,cells+1)
    A = np.array([np.diff(Legendre.basis(j).integ()(edges)) for j in range(m+1)])
    real, imag = cell_integrals(edges,x,y)
    objective = direction[0]*real+direction[1]*imag
    ans = linprog(-objective, A_eq=A, b_eq=np.zeros(m+1),
                  bounds=[(-1,1)]*cells, method='highs',
                  options={'dual_feasibility_tolerance':1e-9,
                           'primal_feasibility_tolerance':1e-9})
    if not ans.success:
        raise RuntimeError(ans.message)
    # min(-c*h) dual multipliers have the opposite sign to our L1 dual.
    lam = -ans.eqlin.marginals
    poly = Legendre(lam).convert(kind=Polynomial).coef
    return {'edges':edges.tolist(), 'values':ans.x.tolist(),
            'lambda_legendre':lam.tolist(), 'lambda_power':poly.tolist(),
            'raw_support':float(-ans.fun/(2*math.pi)), 'm':m,
            'x':x,'y':y,'direction':list(direction),'cells':cells}


def main():
    import argparse,json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--r',type=int,default=1)
    p.add_argument('--height',type=float,required=True)
    p.add_argument('--epsilon',type=float,default=.02)
    p.add_argument('--tolerance',type=float)
    a=p.parse_args()
    out={'r':a.r,'calibration_moments':2*a.r+1,'normalized_height':a.height,
         'log_error_envelope':a.epsilon,
         'unwrapped_phase_radians':phase_budget(a.r,a.height,a.epsilon),
         'log_magnitude_error':gain_budget(a.r,a.height,a.epsilon)}
    if a.tolerance is not None:
        out['minimum_r_float_rule']=required_order(a.height,a.epsilon,a.tolerance)
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
