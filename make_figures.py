from pathlib import Path
import sys,json,math
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'src'))
from phasebudgets import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':11,'axes.spines.top':False,
                     'axes.spines.right':False,'savefig.bbox':'tight','pdf.fonttype':42})
COLORS=['#164b74','#d1622c','#38846b','#8c528a']
F=ROOT/'manuscript/figures';F.mkdir(exist_ok=True)
def save(fig,name):
 fig.savefig(F/(name+'.pdf'));fig.savefig(F/(name+'.png'),dpi=180);plt.close(fig)

y=np.logspace(-4,.4,400)
fig,ax=plt.subplots(1,2,figsize=(8,3.4))
for r,c in zip((0,1,3,7),COLORS):
 ax[0].loglog(y,[phase_budget(r,v) for v in y],color=c,label=f'r = {r}')
 ax[1].loglog(y,[gain_budget(r,v) for v in y],color=c)
for a in ax:
 a.set_xlabel('Normalized height y');a.grid(alpha=.2,which='both')
ax[0].set_ylabel('Sharp unwrapped phase / ε');ax[1].set_ylabel('Sharp log-magnitude / ε')
ax[0].legend(frameon=False);ax[0].set_ylim(1e-10,5);ax[1].set_ylim(1e-10,1)
fig.tight_layout();save(fig,'budgets')

fig,ax=plt.subplots(1,2,figsize=(8,3.1))
for a,kind in zip(ax,('phase','gain')):
 edges,v=extremizer(3,kind);a.stairs(v,edges,baseline=None,color=COLORS[0],linewidth=1.6)
 a.axhline(0,c='grey',lw=.5);a.set_xlim(-1,1);a.set_ylim(-1.2,1.2)
 a.set_xlabel('Band coordinate t');a.set_ylabel('h(t) / ε');a.set_title(kind.capitalize()+' extremizer (r = 3)')
fig.tight_layout();save(fig,'extremizers')

fig,axes=plt.subplots(1,2,figsize=(8,3.8))
for ax,x in zip(axes,(0,.35)):
 for m,c in zip((2,4),COLORS):
  coords=[]
  for th in np.linspace(0,2*np.pi,97):
   p=optimize_cells(m,x,.1,(float(np.cos(th)),float(np.sin(th))),512)
   re,im=cell_integrals(p['edges'],x,.1)
   coords.append([np.dot(re,p['values'])/(2*np.pi),np.dot(im,p['values'])/(2*np.pi)])
  coords=np.array(coords);ax.fill(coords[:,0],coords[:,1],alpha=.13,color=c)
  ax.plot(coords[:,0],coords[:,1],c=c,label=f'm = {m}')
  if x==0:
   r=m//2;A=gain_budget(r,.1);B=phase_budget(r,.1)
   ax.add_patch(Rectangle((-A,-B),2*A,2*B,fill=False,edgecolor=c,linestyle='--',alpha=.7))
 ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('Real log-factor error / ε')
 ax.set_ylabel('Imaginary log-factor error / ε');ax.set_title(f'z = {x:g} + 0.1i');ax.legend(frameon=False)
 ax.grid(alpha=.2)
fig.tight_layout();save(fig,'joint_sets')

smooth=json.loads((ROOT/'results/smooth_tests.json').read_text())
fig,ax=plt.subplots(1,2,figsize=(8,3.5))
deltas=np.linspace(0,1,250)
for y,c in zip((.01,.1,.5),COLORS):
 ax[0].plot(deltas,[noisy_first_moment_budget(y,d) for d in deltas],color=c,label=f'y = {y}')
ax[0].set_xlabel('Allowed first-moment error δ / ε');ax[0].set_ylabel('Sharp phase / ε');ax[0].legend(frameon=False)
for r,c in zip((1,3),COLORS):
 for y,ls in ((.01,'-'),(.1,'--')):
  rr=[v for v in smooth if v['r']==r and v['y']==y]
  ax[1].semilogx([v['kappa'] for v in rr],[v['attainment'] for v in rr],ls+'o',color=c,label=f'r={r}, y={y}')
ax[1].set_xlabel('Smoothing parameter κ');ax[1].set_ylabel('Attained phase / sharp budget');ax[1].legend(frameon=False,fontsize=8)
for a in ax:a.grid(alpha=.2)
fig.tight_layout();save(fig,'robustness')

# Tables are generated from the same implementation as the figures.
s='\\begin{tabular}{rrrrr}\n\\toprule\n$r$ & Moments & $B_r(0.01)/\\epsilon$ & $B_r(0.1)/\\epsilon$ & $A_r(0.1)/\\epsilon$\\\\\n\\midrule\n'
for r in (0,1,3,7,12):s+=f'{r} & {2*r+1} & {phase_budget(r,.01):.6f} & {phase_budget(r,.1):.6f} & {gain_budget(r,.1):.6f}\\\\\n'
s+='\\bottomrule\n\\end{tabular}\n';(ROOT/'manuscript/tables/budget_table.tex').write_text(s)
rows=json.loads((ROOT/'results/continuum_brackets.json').read_text())
s='\\begin{tabular}{rrrrrr}\n\\toprule\n$m$ & $x$ & $y$ & Direction & Lower & Upper\\\\\n\\midrule\n'
for v in rows:
 if v['direction']==[0,1] or v['x']==.35 and v['y']==.1 and v['direction']==[1,1]:
  q=v['certificate'];s+=f"{v['m']} & {v['x']} & {v['y']} & $({v['direction'][0]},{v['direction'][1]})$ & {math.floor(q['lower']*1e6)/1e6:.6f} & {math.ceil(q['upper']*1e6)/1e6:.6f}\\\\\n"
s+='\\bottomrule\n\\end{tabular}\n';(ROOT/'manuscript/tables/bracket_table.tex').write_text(s)
print('Generated four figures and two tables.')
