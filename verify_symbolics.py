"""Algebraic and off-axis checks independent of the centre budget routines."""
from pathlib import Path
import json
import sympy as s
import mpmath as m
x,y=s.symbols('x y',real=True); rows=[]
for r in range(7):
 for kind,n in [('phase',2*r+1),('gain',2*r+2)]:
  U=s.chebyshevu(n,x)
  c=s.cancel((s.I*y if kind=='phase' else y)/U.subs(x,s.I*y))
  rem=s.cancel((x if kind=='phase' else y)-c*U)
  q,res=s.div(s.Poly(rem,x),s.Poly(x*x+y*y,x))
  assert s.simplify(res.as_expr())==0
  assert q.is_zero or q.degree()<=2*r
  rows.append({'r':r,'kind':kind,'quotient_degree':str(q.degree()),'remainder':str(res.as_expr())})
m.mp.dps=65; maxerr=m.mpf(0); numerical=[]
for n in (1,2,3,5,8):
 for xs,ys in [('.3','.2'),('-.6','.07'),('1.4','.4')]:
  z=m.mpc(xs,ys)
  w=z-m.sqrt(z*z-1)
  if abs(w)>1:w=z+m.sqrt(z*z-1)
  edges=[-m.mpf(1)]+sorted([m.cos(k*m.pi/(n+1)) for k in range(1,n+1)])+[m.mpf(1)]
  val=sum(m.sign(m.chebyu(n,(a+b)/2))*m.quad(lambda t:1/(t-z),[a,b]) for a,b in zip(edges[:-1],edges[1:]))
  law=-2*m.log((1+w**(n+1))/(1-w**(n+1)))
  error=abs(val-law);maxerr=max(error,maxerr)
  assert error<m.mpf('1e-58'),(n,z,error)
  numerical.append({'n':n,'x':xs,'y':ys,'error':str(error)})
out={'polynomial_division_identities':rows,'off_axis_transform_checks':numerical,'max_transform_error':str(maxerr)}
(Path(__file__).parent/'results/symbolic_and_off_axis.json').write_text(json.dumps(out,indent=2)+'\n')
print('14 symbolic divisions and 15 off-axis transform checks passed; max error',maxerr)
