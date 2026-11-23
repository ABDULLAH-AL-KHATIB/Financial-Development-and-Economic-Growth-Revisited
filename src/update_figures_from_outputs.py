"""Rebuild the manuscript-facing figures from the frozen, supplied result tables."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parents[1]
T=ROOT/'tables'; F=ROOT/'figures'
BLUE='#0072B2'; ORANGE='#D55E00'; GREEN='#009E73'; GREY='#555555'
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
def save(fig,name):
    for ext in ('png','pdf'):
        fig.savefig(F/f'{name}.{ext}',bbox_inches='tight')
    plt.close(fig)

# F1: correct the supplied benchmark label. The reference contains country and year FE, no trend.
d=pd.read_csv(T/'T10_forecast_ranking.csv')
d['model']=d.model.replace({'FE+trend (reference)':'FE benchmark (two-way FE)'})
d=d.sort_values('RMSE',ascending=True)
fig,ax=plt.subplots(figsize=(6.4,3.2))
colors=[ORANGE if x=='ElasticNet' else (GREY if x=='FE benchmark (two-way FE)' else BLUE) for x in d.model]
ax.barh(d.model,d.RMSE,color=colors)
ax.set_xlabel('Holdout RMSE (2017–2021)')
ax.set_title('Out-of-time forecast comparison')
ax.invert_yaxis(); ax.grid(axis='x',alpha=.18)
save(fig,'F1_ranking')

# F2 is already correctly labelled and uses the five inferential schemes from T03.

# F3 annual loss decomposition; clarify the comparator explicitly.
d=pd.read_csv(T/'T13_tail.csv')
fig,axes=plt.subplots(1,2,figsize=(7.2,3.0))
axes[0].bar(d.year.astype(str),d.dMSE,color=[ORANGE if x>0 else BLUE for x in d.dMSE])
axes[0].axhline(0,color='black',lw=.8)
axes[0].set_title('Annual squared-loss gap')
axes[0].set_ylabel('ElasticNet loss − FE loss')
axes[1].plot(d.year,d.contrib.cumsum(),marker='o',color=BLUE)
axes[1].axhline(0,color='black',ls='--',lw=.8)
axes[1].set_title('Cumulative test loss difference')
fig.tight_layout(); save(fig,'F3_tail')

# F4 leave-one-country-out sensitivity.
jk=pd.read_csv(T/'T06_jackknife.csv'); jc=jk[jk.type.eq('country')].sort_values('beta')
sel=pd.concat([jc.head(5),jc.tail(5)])
fig,ax=plt.subplots(figsize=(6.4,3.1))
ax.barh(sel['drop'],sel.beta,color=BLUE)
ax.axvline(-1.3775736934,color=ORANGE,ls='--',label='Full-sample estimate')
ax.set_xlabel('FD coefficient after dropping one country')
ax.set_title('Leave-one-country-out sensitivity')
ax.legend(frameon=False,fontsize=8); fig.tight_layout(); save(fig,'F4_jackknife')

# F5: remove the misleading vertical line at an unsupported threshold. Show the quadratic component only.
# Values are reconstructed from the stored quadratic coefficient and primary linear estimate.
fd_beta=-1.3775736934091791
quad=-0.19882635218677072
fd=np.linspace(-0.8,2.7,160)
fig,ax=plt.subplots(figsize=(5.8,3.2))
ax.plot(fd,fd_beta*fd+quad*fd**2,color=BLUE,lw=2)
ax.axhline(0,color='black',lw=.8)
ax.set_xlabel('Lagged financial-development index (SD units)')
ax.set_ylabel('Linear + quadratic fitted component (pp)')
ax.set_title('Exploratory quadratic specification')
ax.text(.03,.04,'Quadratic term: p = 0.289\nThreshold bootstrap: p = 0.498\nNo statistically supported threshold.',transform=ax.transAxes,fontsize=8,va='bottom',bbox={'facecolor':'white','edgecolor':'0.8','alpha':0.9})
fig.tight_layout(); save(fig,'F5_nonlinearity')

# F6: use supplied prediction scatter but correct the erroneous "FE+trend" title.
p=F/'F6_scatter.png'
im=Image.open(p).convert('RGB')
draw=ImageDraw.Draw(im)
font_candidates=['/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf','/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf']
font_path=next((x for x in font_candidates if Path(x).exists()),None)
font=ImageFont.truetype(font_path,42) if font_path else ImageFont.load_default()
# The old title sits in the top margin over the left panel; white-out only that title strip.
draw.rectangle((245,0,860,74),fill='white')
draw.text((300,20),'FE benchmark',font=font,fill='black')
im.save(p,dpi=(300,300))
im.save(F/'F6_scatter.pdf','PDF',resolution=300.0)

# F8 title/axis wording: this is the Holm adjustment for the country-mean squared-loss procedure only.
p=F/'F8_holm.png'
im=Image.open(p).convert('RGB'); draw=ImageDraw.Draw(im)
font=ImageFont.truetype(font_path,34) if font_path else ImageFont.load_default()
# Overlay original plot title at the top.
draw.rectangle((340,0,1470,62),fill='white')
draw.text((360,12),'Holm-adjusted country-mean squared-loss tests',font=font,fill='black')
im.save(p,dpi=(300,300)); im.save(F/'F8_holm.pdf','PDF',resolution=300.0)
print('Updated manuscript-facing figures in',F)
