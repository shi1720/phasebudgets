"""Independent checks of assumptions, sharpness and continuum certification."""
import sys,unittest,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from scipy.integrate import quad
from scipy.optimize import linprog
from scipy.special import eval_legendre
from phasebudgets import *
from certifier import certify

class ResearchTests(unittest.TestCase):
    def test_exact_low_order_moments(self):
        # r=1 switch points +-1/sqrt(2),0 preserve degrees 0,1,2.
        a=math.sqrt(.5)
        edges=[-1,-a,0,a,1]; vals=[1,-1,1,-1]
        for j in range(3):
            eta=sum(v*(b**(j+1)-a0**(j+1))/(j+1) for a0,b,v in zip(edges[:-1],edges[1:],vals))
            self.assertLess(abs(eta),1e-14)
    def test_closed_formula_against_continuum_bracket(self):
        for r in (0,1,2):
            for d,fun in [((0,1),phase_budget),((1,0),gain_budget)]:
                cert=certify(optimize_cells(2*r,0,.2,d,128))
                exact=fun(r,.2)
                self.assertLessEqual(cert['lower'],exact)
                self.assertGreaterEqual(cert['upper'],exact)
                self.assertLess(cert['upper']-cert['lower'],.003)
    def test_noisy_formula_against_independent_lp(self):
        cells=4096; edges=np.linspace(-1,1,cells+1)
        mom=np.array([np.diff(edges**(j+1))/(j+1) for j in range(3)])
        for delta in (0.,.1,.5,1.):
            objective=.5*np.diff(np.log(edges**2+.2**2))
            ans=linprog(-objective,A_eq=mom[[0,2]],b_eq=[0,0],
                        A_ub=np.vstack([mom[1],-mom[1]]),b_ub=[delta,delta],
                        bounds=[(-1,1)]*cells,method='highs')
            self.assertTrue(ans.success)
            self.assertAlmostEqual(-ans.fun/(2*math.pi),noisy_first_moment_budget(.2,delta),delta=1e-6)
    def test_scaling_and_order(self):
        for y in (.01,.1,1):
            for tau in (.001,.01,.1):
                r=required_order(y,.02,tau)
                self.assertLessEqual(phase_budget(r,y,.02),tau*(1+1e-12))
                if r: self.assertGreater(phase_budget(r-1,y,.02),tau)
    def test_off_centre_certificates_and_precision(self):
        from flint import ctx
        for d in ((1,0),(0,1),(1,1),(-1,2)):
            p=optimize_cells(4,.35,.1,d,128)
            ctx.prec=192; a=certify(p)
            ctx.prec=384; b=certify(p)
            self.assertAlmostEqual(a['lower'],b['lower'],places=14)
            self.assertAlmostEqual(a['upper'],b['upper'],places=14)
            self.assertGreaterEqual(a['upper'],a['lower'])
    def test_boundary_law(self):
        for r in (0,2,10):
            y=1e-7
            leading=-math.log((r+1)*y)/math.pi
            self.assertAlmostEqual(phase_budget(r,y),leading,places=8)
    def test_extreme_small_height(self):
        y=1e-14
        self.assertAlmostEqual(phase_budget(0,y),-math.log(y)/math.pi,places=12)
        r=required_order(.1,.02,1e-15)
        self.assertLessEqual(phase_budget(r,.1,.02),1e-15*(1+1e-12))

    def test_input_contract(self):
        for args in [(-1,.2),(1,0),(1,-1),(.5,.2)]:
            with self.assertRaises(ValueError):phase_budget(*args)

if __name__=='__main__':unittest.main()
