"""Run all deterministic experiments. See results/summary.json for actual counts."""
from pathlib import Path
import sys,json,time,platform,math
sys.path.insert(0,str(Path(__file__).parent/'src'))
import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import eval_legendre,eval_chebyu
import mpmath as mp
from flint import arb,ctx
from phasebudgets import *
from certifier import certify
ROOT=Path(__file__).parent
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)


def dump(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2)+'\n')


def ball_checks():
    rows=[]; moments=0
    for bits in (192,384):
        ctx.prec=bits
        for r in range(13):
            for ys in ('0.001','0.01','0.1','0.3','1','3'):
                y=arb(ys); aa=y.asinh()
                for kind,n in [('phase',2*r+1),('gain',2*r+2)]:
                    edges=[arb(-1)]+[(arb(k)*arb.pi()/(n+1)).cos() for k in range(n,0,-1)]+[arb(1)]
                    # signs alternate, with sign determined at t near zero.
                    signs=[(-1)**(r if kind=='phase' else r+1)]
                    # U_n has leading coefficient positive, so leftmost sign is (-1)^n.
                    signs=[(-1)**(r if kind=='phase' else r+1)*(-1)**(n+j) for j in range(n+1)]
                    val=arb(0)
                    for a,b,s in zip(edges[:-1],edges[1:],signs):
                        if kind=='phase':
                            val+=s*((b*b+y*y).log()-(a*a+y*y).log())/2
                        else:
                            val+=s*((b/y).atan()-(a/y).atan())
                    val/=2*arb.pi()
                    if kind=='phase':
                        # Full signed integral above already includes both halves.
                        theory=((r+1)*aa).coth().log()/arb.pi()
                    else: theory=2/arb.pi()*(-(2*r+3)*aa).exp().atan()
                    assert (val-theory).contains(0),(r,ys,kind,val,theory)
                    for j in range(2*r+1):
                        eta=sum((s*(b**(j+1)-a**(j+1))/(j+1) for a,b,s in zip(edges[:-1],edges[1:],signs)),arb(0))
                        assert eta.contains(0),(r,j,eta)
                        moments+=1
                    rows.append(dict(bits=bits,r=r,y=ys,kind=kind,integral=val.str(45),formula=theory.str(45)))
    dump('ball_checks.json',rows)
    return len(rows),moments


def independent_checks():
    mp.mp.dps=70; rows=[]
    for r in (0,1,3,6):
        for ys in ('0.01','0.2','1'):
            y=mp.mpf(ys); n=2*r+1
            edges=[-mp.mpf(1)]+sorted([mp.cos(k*mp.pi/(n+1)) for k in range(1,n+1)])+[mp.mpf(1)]
            v=sum(mp.sign((-1)**r*mp.chebyu(n,(a+b)/2))*mp.quad(lambda t:t/(t*t+y*y),[a,b]) for a,b in zip(edges[:-1],edges[1:]))/(2*mp.pi)
            f=mp.log(mp.coth((r+1)*mp.asinh(y)))/mp.pi
            assert abs(v-f)<mp.mpf('1e-60')
            rows.append(dict(r=r,y=ys,absolute_difference=str(abs(v-f))))
    dump('independent_quadrature.json',rows)
    return len(rows)


def continuum_brackets():
    ctx.prec=192; rows=[]
    for m in (2,4):
        for x,y in ((0,.1),(.35,.1),(.35,.5)):
            for direction in ((1,0),(0,1),(1,1)):
                p=optimize_cells(m,x,y,direction,256)
                start=time.perf_counter(); cert=certify(p)
                rows.append({**p,'certificate':cert,'certificate_seconds':time.perf_counter()-start})
                print('bracket',m,x,y,direction,cert['upper']-cert['lower'],flush=True)
                dump('continuum_brackets.json',rows)
    return rows


def convergence():
    rows=[]
    for cells in (32,64,128,256,512):
        for x,y in ((0,.1),(.35,.1)):
            p=optimize_cells(4,x,y,(0,1),cells)
            cert=certify(p)
            rows.append({'cells':cells,'x':x,'y':y,'certificate':cert})
            print('convergence',cells,x,cert['upper']-cert['lower'],flush=True)
            dump('convergence.json',rows)
    return rows


def smooth_tests():
    rows=[]
    for r in (1,3):
        m=2*r; n=2*r+1
        switches=extremizer(r)[0]
        points=switches[1:-1]
        def integrate(f):
            return quad(f,-1,1,points=points,epsabs=2e-12,epsrel=2e-12,limit=600)[0]
        for kappa in (8,32,128,512):
            def bump(t):
                if abs(t)>=1:return 0.
                return math.exp(-1/(kappa*(1-t*t)))
            def h0(t):return bump(t)*math.tanh(kappa*((-1)**r)*eval_chebyu(n,t))
            G=np.array([[integrate(lambda t,i=i,j=j:bump(t)*eval_legendre(i,t)*eval_legendre(j,t)) for j in range(m+1)] for i in range(m+1)])
            eta=np.array([integrate(lambda t,j=j:eval_legendre(j,t)*h0(t)) for j in range(m+1)])
            c=np.linalg.solve(G,eta); D=np.abs(c).sum()
            def h(t):return (h0(t)-bump(t)*sum(c[j]*eval_legendre(j,t) for j in range(m+1)))/(1+D)
            residual=max(abs(integrate(lambda t,j=j:eval_legendre(j,t)*h(t))) for j in range(m+1))
            for y in (.01,.1):
                phase=integrate(lambda t:t/(t*t+y*y)*h(t))/(2*math.pi)
                rows.append(dict(r=r,kappa=kappa,y=y,attainment=phase/phase_budget(r,y),moment_residual=residual,coefficient_bound=D))
    dump('smooth_tests.json',rows);return rows


def main():
    started=time.perf_counter()
    count,moments=ball_checks(); print('ball checks',count,moments,flush=True)
    independent=independent_checks(); print('independent quadrature',independent,flush=True)
    rows=continuum_brackets(); conv=convergence(); smooth=smooth_tests()
    summary={'ball_formula_checks':count,'ball_moment_checks':moments,'independent_quadratures':independent,
             'continuum_certificates':len(rows),'convergence_certificates':len(conv),'smooth_cases':len(smooth),
             'max_continuum_gap':max(v['certificate']['upper']-v['certificate']['lower'] for v in rows),
             'max_smooth_moment_residual':max(v['moment_residual'] for v in smooth),
             'seconds':time.perf_counter()-started,'python':platform.python_version(),
             'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform()}
    dump('summary.json',summary); print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
