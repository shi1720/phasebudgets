"""Independent continuum brackets: exact rational algebra + Arb integration.

Floating LP output is only a proposal. Every recorded float becomes its decimal
rational value. Primal moments are repaired exactly. Dual real roots are isolated
by SymPy over QQ; Arb bounds the antiderivatives at rational endpoints.
"""
import json
import math
import sympy as S
from flint import arb, fmpq, ctx
ctx.prec = 192
T = S.Symbol('t')
R = lambda x: S.Rational(str(x))


def ball(x):
    x = S.Rational(x)
    return arb(fmpq(int(x.p),int(x.q)))


def peval(poly,t):
    out=arb(0)
    for c in reversed(poly):
        out=out*t+ball(c)
    return out


def coefficients(expr):
    p=S.Poly(expr,T)
    return [p.nth(i) for i in range(max(0,p.degree())+1)] if expr != 0 else [S.Integer(0)]


def primitive_kernel(t,x,y,A,B):
    u=t-ball(x)
    return ball(B)/2*(u*u+ball(y*y)).log()+ball((A+B*x)/y)*(u/ball(y)).atan()


def kernel_polynomial_integral(poly,x,y,A,B,left,right):
    D=(T-x)**2+y*y
    q,rem=S.div(S.Poly(poly*(A+B*T),T),S.Poly(D,T))
    qi=coefficients(S.integrate(q.as_expr(),T))
    a,b=rem.nth(0),rem.nth(1)
    def F(t):
        return peval(qi,t)+primitive_kernel(t,x,y,a,b)
    return F(ball(right))-F(ball(left))


def certify(proposal):
    m=proposal['m']
    if type(m) is not int or m<0:
        raise ValueError('moment degree must be a nonnegative integer')
    x=R(proposal['x']); y=R(proposal['y'])
    if y<=0:
        raise ValueError('evaluation height must be positive')
    if len(proposal['direction'])!=2:
        raise ValueError('direction must have two rational components')
    d0,d1=map(R,proposal['direction'])
    if d0==0 and d1==0:
        raise ValueError('direction must be nonzero')
    A=d0*y+d1*x; B=-d1
    edges=list(map(R,proposal['edges'])); values=list(map(R,proposal['values']))
    if (len(edges)<2 or len(values)!=len(edges)-1 or edges[0]!=-1 or
        edges[-1]!=1 or any(a>=b for a,b in zip(edges[:-1],edges[1:]))):
        raise ValueError('cells must form a strictly increasing partition of [-1,1] with one height per cell')
    dual=list(map(R,proposal['lambda_power']))
    if len(dual)>m+1 and any(c!=0 for c in dual[m+1:]):
        raise ValueError('dual polynomial exceeds the calibrated moment degree')
    # Clip decimal values to the declared box before exact repair.
    values=[max(-1,min(1,v)) for v in values]
    legs=[S.legendre(j,T) for j in range(m+1)]
    moments=[]
    for leg in legs:
        F=S.integrate(leg,T)
        moments.append(sum(v*(F.subs(T,b)-F.subs(T,a)) for a,b,v in zip(edges[:-1],edges[1:],values)))
    cs=[S.Rational(2*j+1,2)*eta for j,eta in enumerate(moments)]
    D=sum(abs(c) for c in cs)
    scale=1/(1+D)
    def repaired_component(Ac,Bc):
        total=arb(0)
        for a,b,v in zip(edges[:-1],edges[1:],values):
            total += ball(v)*(primitive_kernel(ball(b),x,y,Ac,Bc)-primitive_kernel(ball(a),x,y,Ac,Bc))
        for c,leg in zip(cs,legs):
            total -= ball(c)*kernel_polynomial_integral(leg,x,y,Ac,Bc,-1,1)
        return ball(scale)*total/(2*arb.pi())
    re=repaired_component(y,0); im=repaired_component(x,-1)
    low=ball(d0)*re+ball(d1)*im
    # Dual candidate in power basis, treated as exact decimal rationals.
    p=sum(c*T**j for j,c in enumerate(dual))
    numerator=S.Poly(A+B*T-((T-x)**2+y*y)*p,T)
    intervals=numerator.intervals(eps=S.Rational(1,10**25),inf=-1,sup=1)
    boxes=[]
    for (a,b),mult in intervals:
        a=max(a,S.Integer(-1)); b=min(b,S.Integer(1))
        if a <= b: boxes.append((a,b))
    pi=coefficients(S.integrate(p,T))
    def F(t):
        return primitive_kernel(t,x,y,A,B)-peval(pi,t)
    upper=arb(0); last=S.Integer(-1)
    M=sum(abs(c) for c in numerator.all_coeffs())/(y*y)
    for a,b in boxes+[(S.Integer(1),S.Integer(1))]:
        if a>last:
            sig=S.sign(numerator.eval((a+last)/2))
            upper += int(sig)*(F(ball(a))-F(ball(last)))
        if b>a: upper += ball((b-a)*M)
        last=b
    upper /= 2*arb.pi()
    if float(low.lower()) > float(upper.upper())+1e-14:
        raise ArithmeticError('primal exceeds dual')
    return {'lower':math.nextafter(float(low.lower()),-math.inf),'upper':math.nextafter(float(upper.upper()),math.inf),
            'lower_ball':low.str(45),'upper_ball':upper.str(45),
            'real':float(re.mid()),'imag':float(im.mid()),
            'repair_norm':float(ball(D).upper()), 'scale':str(scale),
            'root_intervals':[[str(a),str(b)] for a,b in boxes],
            'precision_bits':ctx.prec}
