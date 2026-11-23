"""Refresh the corrected covariance table and plot after a DK-code audit."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
T=ROOT/'tables'; F=ROOT/'figures'
for fn in ['T03_primary_inference.csv','T16_inflation_outlier_robustness.csv','T18_country_trend_robustness.csv']:
    d=pd.read_csv(T/fn)
    try: (T/fn.replace('.csv','.tex')).write_text(d.to_latex(index=False,float_format='%.4f'))
    except Exception: pass

d=pd.read_csv(T/'T03_primary_inference.csv')
labels={
    'Country-clustered':'Country cluster',
    'CGM two-way':'Two-way cluster',
    'Driscoll-Kraay':'Driscoll–Kraay (corrected)',
    'Wild bootstrap (Rademacher)':'Wild cluster: Rademacher',
    'Wild bootstrap (Webb 6-point)':'Wild cluster: Webb six-point'
}
d['label']=d.inference.map(labels).fillna(d.inference)
# Order: country, two-way, DK, wild Rademacher, wild Webb
order=['Country-clustered','CGM two-way','Driscoll-Kraay','Wild bootstrap (Rademacher)','Wild bootstrap (Webb 6-point)']
d=d.set_index('inference').loc[order].reset_index()
fig,ax=plt.subplots(figsize=(6.5,3.4))
y=range(len(d))
lo=d.coef-d.lo; hi=d.hi-d.coef
colors=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00']
for i,row in d.iterrows():
    ax.errorbar(row.coef,i,xerr=[[lo.iloc[i]],[hi.iloc[i]]],fmt='o',capsize=3,color=colors[i],ecolor=colors[i],markersize=5,lw=1.2)
ax.axvline(0,color='black',lw=.8,ls='--')
ax.set_yticks(list(y)); ax.set_yticklabels(d.label)
ax.invert_yaxis(); ax.set_xlabel('Coefficient on lagged overall FDI (percentage points per SD)')
ax.set_title('Primary fixed-effects estimate: corrected inference')
ax.grid(axis='x',alpha=.18)
fig.tight_layout()
for ext in ['png','pdf']: fig.savefig(F/f'F2_inference.{ext}',bbox_inches='tight')
plt.close(fig)
print('Updated corrected covariance output table and F2 figure.')
