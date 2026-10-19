"""Reject malformed continuum proposals and check extreme scalar evaluation."""
import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from phasebudgets import optimize_cells, noisy_first_moment_budget
from certifier import certify

class CertificateContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proposal=optimize_cells(2,.35,.1,(1,1),32)

    def test_degree_contract(self):
        for m in [-1, True, .5]:
            p=copy.deepcopy(self.proposal);p['m']=m
            with self.subTest(m=m),self.assertRaises(ValueError):certify(p)
        p=copy.deepcopy(self.proposal);p['lambda_power']=[0,0,0,1]
        with self.assertRaises(ValueError):certify(p)

    def test_partition_contract(self):
        variants=[]
        p=copy.deepcopy(self.proposal);p['edges'][0]=-.9;variants.append(p)
        p=copy.deepcopy(self.proposal);p['edges'][2]=p['edges'][1];variants.append(p)
        p=copy.deepcopy(self.proposal);p['values'].pop();variants.append(p)
        p=copy.deepcopy(self.proposal);p['edges']=[];variants.append(p)
        for p in variants:
            with self.assertRaises(ValueError):certify(p)

    def test_domain_and_direction_contract(self):
        for y in [0,-.1]:
            p=copy.deepcopy(self.proposal);p['y']=y
            with self.assertRaises(ValueError):certify(p)
        for direction in [[0,0],[1]]:
            p=copy.deepcopy(self.proposal);p['direction']=direction
            with self.assertRaises(ValueError):certify(p)

    def test_noisy_formula_extreme_heights(self):
        import mpmath as mp
        with mp.workdps(120):
            for ys in ['1e-200','1','1e8']:
                y=mp.mpf(ys)
                for ds in ['0','.1','1']:
                    d=mp.mpf(ds);a2=(1+d)/2
                    exact=mp.log((y*y+a2)**2/(y*y*(1+y*y)))/(2*mp.pi)
                    value=noisy_first_moment_budget(float(y),float(d))
                    self.assertGreater(value,0)
                    self.assertLess(abs(mp.mpf(value)-exact)/exact,mp.mpf('5e-15'))

if __name__=='__main__':unittest.main()
