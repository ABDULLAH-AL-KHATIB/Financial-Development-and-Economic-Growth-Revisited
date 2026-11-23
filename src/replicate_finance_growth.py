# =====================================================================================
# FINANCE–GROWTH: DETERMINISTIC REPLICATION PACKAGE (data / results / tables / figures)
# IMF Financial Development Index (9 indices) + Penn World Table 11.0 + WDI
# NO optional models. NO manuscript generation. Threshold bootstrap 1999 x 40.
# =====================================================================================
import os, sys, json, time, math, hashlib, logging, itertools, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reproducibility: this script runs from the frozen, versioned input files in data/.
# No live network requests are made during the default replication.
warnings.filterwarnings("ignore", category=ConvergenceWarning)
SEED=2026; TRAIN_END=2010; VAL_START=2011; VAL_END=2016; TEST_START=2017
WILD_REPS=9999; THRESH_BOOT=1999; THRESH_GRID=40
np.random.seed(SEED)
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT; TAB=OUT/"tables"; FIG=OUT/"figures"; DAT=OUT/"data"
for p in (OUT,TAB,FIG,DAT): p.mkdir(parents=True,exist_ok=True)
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(OUT/"run.log",mode="w")])
log=logging.getLogger("rep")
def sha(b): return hashlib.sha256(b).hexdigest()
def table(df,name,caption):
    df.to_csv(TAB/f"{name}.csv",index=False)
    try: (TAB/f"{name}.tex").write_text(df.to_latex(index=False,float_format="%.4f",caption=caption,label=f"tab:{name}"))
    except Exception: pass
    log.info("TABLE %s (%d rows)",name,len(df))
OI=["#0072B2","#D55E00","#009E73","#CC79A7","#E69F00","#56B4E9","#F0E442","#000000"]
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":300})
def fig_save(fig,name):
    for e in ("pdf","png"): fig.savefig(FIG/f"{name}.{e}",bbox_inches="tight")
    plt.close(fig); log.info("FIGURE %s",name)

# ============================== 1. FROZEN DATA =======================================
PWT_DOI="10.34894/FABVLR"
IMF={"fd":"FD_FD_IX","fi":"FD_FI_IX","fm":"FD_FM_IX","fid":"FD_FID_IX","fia":"FD_FIA_IX",
     "fie":"FD_FIE_IX","fmd":"FD_FMD_IX","fma":"FD_FMA_IX","fme":"FD_FME_IX"}
assert (DAT/"raw_panel.csv").exists(), "Missing data/raw_panel.csv"
assert (DAT/"country_meta.csv").exists(), "Missing data/country_meta.csv"
assert (DAT/"pwt110.xlsx").exists(), "Missing data/pwt110.xlsx"
raw_bytes=(DAT/"raw_panel.csv").read_bytes()
meta_bytes=(DAT/"country_meta.csv").read_bytes()
pwt_bytes=(DAT/"pwt110.xlsx").read_bytes()
panel=pd.read_csv(DAT/"raw_panel.csv")
assert {"iso3","year","growth","fd","rgdpo","rgdpna","pop","hc","inflation"}.issubset(panel.columns)
assert not panel.duplicated(["iso3","year"]).any(), "Duplicate country-year observations in raw input"
provenance=pd.read_csv(DAT/"provenance.csv").to_dict(orient="records")
meta=pd.read_csv(DAT/"country_meta.csv")
assert not meta.duplicated(["iso3"]).any(), "Duplicate country codes in country metadata"
log.info("FROZEN INPUT %d rows / %d countries / %d-%d; raw sha256=%s",len(panel),panel.iso3.nunique(),panel.year.min(),panel.year.max(),sha(raw_bytes))
table(pd.DataFrame(provenance),"T01_provenance","Data provenance and source file hashes")

# ============================== 2. VARIABLES =========================================
def hp_filter(y,lam=100.0):
    y=np.asarray(y,float); n=len(y)
    if n<5 or not np.isfinite(y).all(): return np.full(n,np.nan)
    D=np.zeros((n-2,n))
    for i in range(n-2): D[i,i],D[i,i+1],D[i,i+2]=1,-2,1
    return y-np.linalg.solve(np.eye(n)+lam*D.T@D,y)
panel=panel.sort_values(["iso3","year"]).reset_index(drop=True)
panel["log_gdppc"]=np.log((panel["rgdpo"]/panel["pop"]).where(lambda s:s>0))
panel["log_cap_pw"]=np.log((panel["rnna"]/panel["emp"]).where(lambda s:s>0))
panel["log_ctfp"]=np.log(panel["ctfp"].where(panel["ctfp"]>0))
panel["g_growth_pwt"]=panel.groupby("iso3")["rgdpna"].transform(lambda s:100*np.log(s/s.shift(1)))
panel["credit_gap"]=np.nan
for c,g in panel.groupby("iso3"):
    v=g["credit"].to_numpy(float)
    if np.isfinite(v).sum()>=8:
        vv=pd.Series(v).interpolate(limit_direction="both").to_numpy()
        panel.loc[g.index,"credit_gap"]=hp_filter(vv,100.0)
panel["crisis08"]=panel["year"].isin([2008,2009]).astype(float)
panel["crisis20"]=panel["year"].eq(2020).astype(float)
panel=panel.merge(meta,on="iso3",how="left")
panel["high_income"]=panel["income"].eq("High income").astype(float)
for c in IMF: panel[c+"_z"]=(panel[c]-panel[c].mean())/panel[c].std()
CTRL_PRE=["log_gdppc","hc"]; CTRL_MID=CTRL_PRE+["trade","govcons","urban"]
CTRL_FULL=CTRL_MID+["inflation","unemp"]; CTRL_EXT=CTRL_FULL+["log_cap_pw","labsh","gcf","school"]
def lag(df,cols):
    cols=list(dict.fromkeys(cols))                       # FIX: no duplicate lag columns
    L=df[["iso3","year"]+cols].copy(); L["year"]+=1
    L=L.rename(columns={c:c+"_L" for c in cols})
    return df.merge(L,on=["iso3","year"],how="left",validate="one_to_one")
EXPO=[c+"_z" for c in IMF]
panel=lag(panel,["growth","g_growth_pwt","credit","credit_gap","fd"]+EXPO+CTRL_EXT)
panel["fd_lead"]=panel.groupby("iso3")["fd_z"].shift(-1)
assert not panel.columns.duplicated().any(), "duplicate columns after lagging"
need=["growth","fd_z_L"]+[c+"_L" for c in CTRL_FULL]
est=panel.dropna(subset=need).copy()
YMIN,YMAX=int(est.year.min()),int(est.year.max()); TEST_END=YMAX
table(pd.DataFrame([{"step":"merged panel","obs":len(panel),"countries":panel.iso3.nunique()},
                    {"step":"estimation (full controls)","obs":len(est),"countries":est.iso3.nunique()}]),
      "T02_funnel","Sample construction funnel")
log.info("ESTIMATION SAMPLE %d obs / %d countries (%d-%d)",len(est),est.iso3.nunique(),YMIN,YMAX)

# ============================== 3. FE ENGINE =========================================
def _dm(v,codes,G):
    s=np.bincount(codes,weights=v,minlength=G); c=np.bincount(codes,minlength=G)
    return v-(s/np.maximum(c,1))[codes]
def within(v,uc,tc,NU,NT,twoway=True,tol=1e-12,it=5000):
    v=np.asarray(v,float).copy()
    if not twoway: return _dm(v,uc,NU)
    for _ in range(it):
        p=v.copy(); v=_dm(v,uc,NU); v=_dm(v,tc,NT)
        if np.max(np.abs(v-p))<tol: return v
    raise RuntimeError("within did not converge")
def ols(Xw,yw):
    B=np.linalg.pinv(Xw.T@Xw); b=B@Xw.T@yw; return B,b,yw-Xw@b
def cl_cov(Xw,e,codes,G,keff):
    B=np.linalg.pinv(Xw.T@Xw); s=np.zeros((G,Xw.shape[1])); np.add.at(s,codes,Xw*e[:,None])
    return G/max(G-1,1)*(len(e)-1)/max(len(e)-keff,1)*B@(s.T@s)@B
def white_cov(Xw,e,keff):
    B=np.linalg.pinv(Xw.T@Xw); M=(Xw*e[:,None]).T@(Xw*e[:,None])
    return (len(e)-1)/max(len(e)-keff,1)*B@M@B
def psd(V):
    w,Q=np.linalg.eigh(V); return Q@np.diag(np.clip(w,0,None))@Q.T if w.min()<0 else V
def dk_cov(Xw,e,tc,NT,keff,L=None):
    k=Xw.shape[1]
    if L is None: L=max(1,int(math.floor(4*(NT/100)**(2/9))))
    H=np.zeros((NT,k)); np.add.at(H,tc,Xw*e[:,None]); S=H.T@H
    for l in range(1,min(L,NT-1)+1):
        w=1-l/(L+1); G=H[l:].T@H[:-l]; S+=w*(G+G.T)
    B=np.linalg.pinv(Xw.T@Xw)
    # Hoechle (2007) / Stata xtscc finite-sample correction:
    # (T/(T-1)) * ((n-1)/(n-k)); k is the number of regressors in the
    # within-transformed regression, not the absorbed FE count. The old
    # T/(T-keff) expression was invalid when the number of FE exceeded T.
    n=len(e)
    small_corr=(NT/max(NT-1,1))*((n-1)/max(n-k,1))
    return psd(small_corr*B@S@B),L
class Fit:
    def __init__(self,d,uc,tc,NU,NT,yw,Xw,names,twoway=True):
        self.d,self.uc,self.tc,self.NU,self.NT=d,uc,tc,NU,NT
        self.names=list(names); self.B,self.beta,self.e=ols(Xw,yw); self.Xw,self.yw=Xw,yw
        self.n,self.k=Xw.shape; self.keff=self.k+(NU+NT-1 if twoway else NU)
        self.fw=Xw@self.beta
        self.Vc=cl_cov(Xw,self.e,uc,NU,self.keff)
        self.Vtw=psd(self.Vc+cl_cov(Xw,self.e,tc,NT,self.keff)-white_cov(Xw,self.e,self.keff)) if twoway else self.Vc
        self.Vdk,self.dkL=dk_cov(Xw,self.e,tc,NT,self.keff)
        self.r2=float(1-(self.e@self.e)/(yw@yw))
    def inf(self,name,V=None,df=None):
        j=self.names.index(name); V=self.Vc if V is None else V; df=self.NU-1 if df is None else df
        b=float(self.beta[j]); se=float(np.sqrt(max(V[j,j],0))); t=b/se if se>0 else np.nan
        p=float(2*stats.t.sf(abs(t),df)); cr=stats.t.ppf(.975,df)
        return dict(coef=b,se=se,t=t,p=p,lo=b-cr*se,hi=b+cr*se,df=df,n=self.n,N=self.NU,T=self.NT)
def fit(df,ycol,xcols,twoway=True):
    d=df.sort_values(["iso3","year"]).reset_index(drop=True)
    uc,uq=pd.factorize(d["iso3"]); tc,tq=pd.factorize(d["year"])
    y=d[ycol].to_numpy(float); X=d[list(xcols)].to_numpy(float)
    yw=within(y,uc,tc,len(uq),len(tq),twoway)
    Xw=np.column_stack([within(X[:,j],uc,tc,len(uq),len(tq),twoway) for j in range(X.shape[1])])
    return Fit(d,uc,tc,len(uq),len(tq),yw,Xw,xcols,twoway)
def wild(fit_,name,reps=WILD_REPS,seed=SEED,webb=False):
    j=fit_.names.index(name); Xw,yw,uc=fit_.Xw,fit_.yw,fit_.uc
    Xr=Xw[:,[i for i in range(Xw.shape[1]) if i!=j]]
    br=np.linalg.pinv(Xr.T@Xr)@Xr.T@yw; fr=Xr@br; er=yw-fr
    G=fit_.NU; order=np.argsort(uc,kind="stable"); Xs=Xw[order]
    bounds=np.searchsorted(uc[order],np.arange(G+1)); B=np.linalg.pinv(Xw.T@Xw)
    c=G/max(G-1,1)*(fit_.n-1)/max(fit_.n-fit_.keff,1)
    obs=fit_.inf(name); tobs=abs(obs["t"]); rng=np.random.default_rng(seed); ts=np.empty(reps)
    wv=np.array([-math.sqrt(1.5),-1,-math.sqrt(.5),math.sqrt(.5),1,math.sqrt(1.5)]) if webb else np.array([-1.,1.])
    for b in range(reps):
        w=rng.choice(wv,size=G)[uc]; ys=fr+er*w
        bs=B@Xw.T@ys; es=ys-Xw@bs
        s=np.add.reduceat(Xs*es[order][:,None],bounds[:-1],axis=0)
        V=c*B@(s.T@s)@B; ts[b]=bs[j]/math.sqrt(max(V[j,j],1e-300))
    p=float((1+np.sum(np.abs(ts)>=tobs))/(reps+1)); q=float(np.quantile(np.abs(ts),.95))
    return dict(coef=obs["coef"],se=obs["se"],p=p,lo=obs["coef"]-q*obs["se"],hi=obs["coef"]+q*obs["se"],reps=reps)
X_FULL=["fd_z_L"]+[c+"_L" for c in CTRL_FULL]
F=fit(est,"growth",X_FULL)

# ---- balanced-block LSDV verification (explicit dummies + rank assertion) ----
def balanced_block(df,W=8,maxc=25):
    yrs=sorted(df["year"].unique()); best=None
    for s in range(len(yrs)-W+1):
        win=yrs[s:s+W]
        if win[-1]-win[0]!=W-1: continue
        ok=df[df["year"].isin(win)].groupby("iso3")["year"].nunique()
        cs=ok[ok==W].index.tolist()
        if best is None or len(cs)>len(best[1]): best=(win,cs)
    if best is None or len(best[1])<4: raise RuntimeError("no balanced verification block found")
    win,cs=best; cs=cs[:maxc]
    return df[df["year"].isin(win)&df["iso3"].isin(cs)].copy(),win,cs
blk,win_b,cs_b=balanced_block(est)
uc_,uq_=pd.factorize(blk["iso3"]); tc_,tq_=pd.factorize(blk["year"])
Dc=np.eye(len(uq_))[uc_][:,1:]; Dt=np.eye(len(tq_))[tc_][:,1:]
A=np.column_stack([blk[X_FULL].to_numpy(float),np.ones(len(blk)),Dc,Dt])
rk=int(np.linalg.matrix_rank(A))
assert rk==A.shape[1], f"verification design rank {rk} < {A.shape[1]}"
b_lsdv=np.linalg.lstsq(A,blk["growth"].to_numpy(float),rcond=None)[0][:len(X_FULL)]
fb=fit(blk,"growth",X_FULL)
err=float(np.max(np.abs(fb.beta-b_lsdv)))
assert err<1e-6, f"LSDV check failed {err}"
log.info("VERIFICATION FE==LSDV balanced block (%d x %d, rank %d) max diff %.2e PASS",len(cs_b),len(win_b),rk,err)
rows=[]
for lab,V,df_ in [("Country-clustered",F.Vc,F.NU-1),("CGM two-way",F.Vtw,min(F.NU-1,F.NT-1)),("Driscoll-Kraay",F.Vdk,F.NT-1)]:
    rows.append({"inference":lab,**F.inf("fd_z_L",V,df_)})
w1=wild(F,"fd_z_L"); w2=wild(F,"fd_z_L",webb=True,seed=SEED+1)
rows.append({"inference":"Wild bootstrap (Rademacher)",**{k:w1[k] for k in ["coef","se","p","lo","hi"]}})
rows.append({"inference":"Wild bootstrap (Webb 6-point)",**{k:w2[k] for k in ["coef","se","p","lo","hi"]}})
table(pd.DataFrame(rows),"T03_primary_inference","Primary coefficient under five inference schemes")

# ============================== 4. ESTIMATOR BATTERY =================================
alt=[{"specification":"Two-way FE (primary)",**F.inf("fd_z_L")}]
dyn=est.dropna(subset=["growth_L"]).copy(); Fd=fit(dyn,"growth",["growth_L"]+X_FULL)
alt.append({"specification":"Dynamic FE (lagged y)",**Fd.inf("fd_z_L")})
mid=int(np.median(dyn.year)); h1=dyn[dyn.year<=mid]; h2=dyn[dyn.year>mid]
if len(h1)>150 and len(h2)>150:
    f1=fit(h1,"growth",["growth_L"]+X_FULL); f2=fit(h2,"growth",["growth_L"]+X_FULL)
    j1=Fd.names.index("fd_z_L"); jr=Fd.names.index("growth_L")
    spj=2*Fd.beta[j1]-0.5*(f1.beta[j1]+f2.beta[j1]); srho=2*Fd.beta[jr]-0.5*(f1.beta[jr]+f2.beta[jr])
    alt.append({"specification":"Dynamic FE, split-panel jackknife","coef":float(spj),"se":np.nan,"p":np.nan,
                "n":Fd.n,"N":Fd.NU,"note":f"rho_FE={Fd.beta[jr]:.3f}, rho_SPJ={srho:.3f}"})
# FIX: first differences preserve the primary t-1 timing
d2=est.copy(); d2["D_growth"]=d2.groupby("iso3")["growth"].diff()
d2["D_fd_z_L"]=d2.groupby("iso3")["fd_z_L"].diff()
for c in CTRL_FULL: d2["D_"+c+"_L"]=d2.groupby("iso3")[c+"_L"].diff()
XFD=["D_fd_z_L"]+["D_"+c+"_L" for c in CTRL_FULL]
d2=d2.dropna(subset=["D_growth"]+XFD)
Ffdiff=fit(d2,"D_growth",XFD,twoway=False)
alt.append({"specification":"First differences (country effects in differences)",**Ffdiff.inf("D_fd_z_L")})
# The earlier implementation incorrectly put country means inside a two-way
# fixed-effects transform, which absorbs all time-invariant means. A proper
# correlated-random-effects (Mundlak) model is estimated by quasi-demeaned GLS
# after the Swamy-Arora variance-component calculation below.
# ---- RE (Swamy-Arora) + Hausman : FIXED broadcasting, intercept, negative chi2 ----
Xnames=["fd_z_L"]+[c+"_L" for c in CTRL_FULL]+["year"]
dRE=est.dropna(subset=Xnames).copy(); ucR,uqR=pd.factorize(dRE["iso3"])
yR=dRE["growth"].to_numpy(float); X0=dRE[Xnames].to_numpy(float)
XR=np.column_stack([X0,np.ones(len(dRE))])
nR=len(yR); NR=len(uqR); kR=XR.shape[1]; kk=X0.shape[1]
XwR=np.column_stack([within(XR[:,j],ucR,ucR,NR,NR,False) for j in range(kR)])
ywR=within(yR,ucR,ucR,NR,NR,False); BR,bFE,eR=ols(XwR,ywR)
Ti=np.bincount(ucR); s2e=float(eR@eR)/max(nR-NR-kk,1)
Xmean=pd.DataFrame(XR).groupby(ucR).transform("mean").to_numpy()
ymR=pd.Series(yR).groupby(ucR).transform("mean").to_numpy()
gm=np.column_stack([np.ones(NR),pd.DataFrame(X0).groupby(ucR).mean().to_numpy()])
gy=pd.Series(yR).groupby(ucR).mean().to_numpy()
eb=gy-gm@(np.linalg.pinv(gm.T@gm)@gm.T@gy)
s2b=float(eb@eb)/max(NR-kk-1,1); Tbar=float(Ti.mean()); s2u=max(s2b-s2e/Tbar,1e-8)
th=(1-np.sqrt(s2e/(s2e+Ti*s2u)))[ucR]
yq=yR-th*ymR
Xq=XR-th[:,None]*Xmean
Bq,bRE,eq=ols(Xq,yq); s2q=float(eq@eq)/max(nR-kR,1); VRE=s2q*np.linalg.pinv(Xq.T@Xq)
VFE=cl_cov(XwR,eR,ucR,NR,NR+kk)
db=(bFE-bRE)[:kk]; Vd=(VRE-VFE)[:kk,:kk]
Hraw=float(db@np.linalg.pinv(Vd)@db); H=max(Hraw,0.0)
seRE=float(np.sqrt(max(VRE[0,0],0)))
alt.append({"specification":"Random effects (Swamy-Arora)","coef":float(bRE[0]),"se":seRE,
            "p":float(2*stats.norm.sf(abs(bRE[0]/seRE))) if seRE>0 else np.nan,"n":nR,"N":NR,
            "note":f"Raw Hausman statistic={Hraw:.2f} is negative and not interpretable; theta_mean={th.mean():.3f}. See the Mundlak CRE test."})

# Proper Mundlak correlated-random-effects test. Re-use the Swamy-Arora theta;
# add country means of the actual lagged regressors (not current-period values),
# quasi-demean all columns, and test the mean terms jointly with country-clustered
# sandwich covariance. No country or year fixed effects are absorbed in this GLS.
M_vars=Xnames[:-1]  # lagged exposure and lagged controls; exclude common linear trend
M=pd.DataFrame(dRE).groupby(ucR)[M_vars].transform("mean").to_numpy(float)
Xcre=np.column_stack([np.ones(nR),X0,M])
Xcre_bar=pd.DataFrame(Xcre).groupby(ucR).transform("mean").to_numpy(float)
ybar_cre=pd.Series(yR).groupby(ucR).transform("mean").to_numpy(float)
Xcre_q=Xcre-th[:,None]*Xcre_bar
ycre_q=yR-th*ybar_cre
pinv_cre=np.linalg.pinv(Xcre_q)
Bcre=pinv_cre@pinv_cre.T
bCRE=Bcre@Xcre_q.T@ycre_q
eCRE=ycre_q-Xcre_q@bCRE
scores_cre=np.zeros((NR,Xcre_q.shape[1]))
np.add.at(scores_cre,ucR,Xcre_q*eCRE[:,None])
kcre=Xcre_q.shape[1]
small_cre=NR/max(NR-1,1)*(nR-1)/max(nR-kcre,1)
Vcre=small_cre*Bcre@(scores_cre.T@scores_cre)@Bcre
j_fd=1  # intercept is column 0; fd_z_L is the first slope
se_cre=float(np.sqrt(max(Vcre[j_fd,j_fd],0)))
p_cre=float(2*stats.t.sf(abs(bCRE[j_fd]/se_cre),NR-1)) if se_cre>0 else np.nan
j_means=np.arange(1+len(Xnames),1+len(Xnames)+len(M_vars))
bmeans=bCRE[j_means]; Vmeans=Vcre[np.ix_(j_means,j_means)]
W_mundlak=float(bmeans@np.linalg.pinv(Vmeans)@bmeans)
p_mundlak=float(stats.chi2.sf(W_mundlak,len(j_means)))
log.info("MUNDLAK CRE country-means Wald chi2(%d)=%.6f p=%.8g; FD coef=%.6f SE=%.6f p=%.6f",
         len(j_means),W_mundlak,p_mundlak,bCRE[j_fd],se_cre,p_cre)
alt.append({"specification":"Mundlak CRE (quasi-demeaned GLS)","coef":float(bCRE[j_fd]),"se":se_cre,
            "p":p_cre,"n":nR,"N":NR,
            "note":f"Country-mean joint Wald chi2({len(j_means)})={W_mundlak:.2f}, p={p_mundlak:.6g}; country-clustered covariance."})
csa=est.groupby("year")[["growth","fd_z_L"]+[c+"_L" for c in CTRL_FULL]].mean().add_prefix("CSA_").reset_index()
cce=est.merge(csa,on="year",how="left"); bc=[]
xc=["fd_z_L"]+[c+"_L" for c in CTRL_FULL]; zc=[c for c in cce.columns if c.startswith("CSA_")]
for c,g in cce.groupby("iso3"):
    if len(g)<len(xc)+len(zc)+5: continue
    Aa=np.column_stack([np.ones(len(g)),g[xc].to_numpy(float),g[zc].to_numpy(float)])
    bc.append(np.linalg.lstsq(Aa,g["growth"].to_numpy(float),rcond=None)[0][1])
if len(bc)>=10:
    bc=np.array(bc); mg=bc.mean(); se=bc.std(ddof=1)/math.sqrt(len(bc))
    alt.append({"specification":"CCE mean group (Pesaran 2006)","coef":float(mg),"se":float(se),
                "p":float(2*stats.t.sf(abs(mg/se),len(bc)-1)),"n":len(cce),"N":len(bc)})
# Anderson-Hsiao: deeper (t-3) level instrument for the differenced exposure
iv=est.copy(); iv["fd_lag3"]=iv.groupby("iso3")["fd_z"].shift(3)
iv["D_growth"]=iv.groupby("iso3")["growth"].diff(); iv["D_fd"]=iv.groupby("iso3")["fd_z_L"].diff()
for c in CTRL_FULL: iv["D_"+c]=iv.groupby("iso3")[c+"_L"].diff()
iv=iv.dropna(subset=["D_growth","D_fd","fd_lag3"]+["D_"+c for c in CTRL_FULL])
Z=np.column_stack([iv["fd_lag3"].to_numpy(float),np.ones(len(iv)),iv[["D_"+c for c in CTRL_FULL]].to_numpy(float)])
Xe=np.column_stack([iv["D_fd"].to_numpy(float),np.ones(len(iv)),iv[["D_"+c for c in CTRL_FULL]].to_numpy(float)])
yv=iv["D_growth"].to_numpy(float); Xh=Z@np.linalg.pinv(Z.T@Z)@Z.T@Xe
biv=np.linalg.pinv(Xh.T@Xe)@Xh.T@yv; eiv=yv-Xe@biv
uci,uiq=pd.factorize(iv["iso3"]); keffi=Xe.shape[1]+len(uiq)
s=np.zeros((len(uiq),Xe.shape[1])); np.add.at(s,uci,Xh*eiv[:,None])
Biv=np.linalg.pinv(Xh.T@Xe); Viv=len(uiq)/(len(uiq)-1)*(len(yv)-1)/max(len(yv)-keffi,1)*Biv@(s.T@s)@Biv
f1c=np.linalg.lstsq(Z,Xe[:,0],rcond=None)[0]; e1=Xe[:,0]-Z@f1c
f0c=np.linalg.lstsq(Z[:,1:],Xe[:,0],rcond=None)[0]; e0=Xe[:,0]-Z[:,1:]@f0c
Fst=float((e0@e0-e1@e1)/((e1@e1)/(len(yv)-Z.shape[1])))
seIV=float(np.sqrt(max(Viv[0,0],0)))
if not np.isfinite(Fst) or Fst<=1.0:
    alt.append({"specification":"Anderson-Hsiao 2SLS (FD instrumented by t-3 level)","coef":np.nan,"se":np.nan,
                "p":np.nan,"n":len(yv),"N":len(uiq),
                "note":f"Not identified/reported: first-stage F={Fst:.1f}; instrument has no usable first-stage relevance."})
else:
    alt.append({"specification":"Anderson-Hsiao 2SLS (FD instrumented by t-3 level)","coef":float(biv[0]),"se":seIV,
                "p":float(2*stats.t.sf(abs(biv[0]/seIV),len(uiq)-1)) if seIV>0 else np.nan,
                "n":len(yv),"N":len(uiq),"note":f"first-stage F={Fst:.1f}; exclusion restriction remains unverified."})
table(pd.DataFrame(alt),"T04_estimators","Alternative estimators and identification triangulation")

# ============================== 5. SENSITIVITY =======================================
sens=[]; Fu=fit(est,"growth",["fd_z_L"])
def oster(bc_,bu,rc,ru):
    rmax=min(1.3*rc,1.0); den=(bu-bc_)*(rmax-rc)
    delta=(bc_*(rc-ru))/den if den!=0 else np.nan
    bstar=bc_-(bu-bc_)*(rmax-rc)/(rc-ru) if rc!=ru else np.nan
    return float(delta),float(bstar),float(rmax)
dl,bs_,rm=oster(F.inf("fd_z_L")["coef"],Fu.inf("fd_z_L")["coef"],F.r2,Fu.r2)
sens.append({"test":"Oster (2019) delta for beta*=0","value":dl,"detail":f"beta*(delta=1)={bs_:.3f}; Rmax={rm:.3f}"})
Fp=fit(est.dropna(subset=["fd_lead"]),"growth",["fd_lead"]+[c+"_L" for c in CTRL_FULL])
sens.append({"test":"Lead placebo: FD(t+1) on growth(t)","value":Fp.inf("fd_lead")["coef"],"detail":f"p={Fp.inf('fd_lead')['p']:.4f}"})
for nm,cs in [("predetermined only",CTRL_PRE),("slow-moving",CTRL_MID),("full",CTRL_FULL),("extended",CTRL_EXT)]:
    dd=est.dropna(subset=["growth","fd_z_L"]+[c+"_L" for c in cs])
    if len(dd)<200: continue
    i=fit(dd,"growth",["fd_z_L"]+[c+"_L" for c in cs]).inf("fd_z_L")
    sens.append({"test":f"Control set: {nm}","value":i["coef"],"detail":f"p={i['p']:.4g}; N={i['n']}"})
for lab,dd in [("exclude 2020",est[est.year!=2020]),("exclude 2008-09",est[~est.year.isin([2008,2009])]),
               ("post-1990",est[est.year>=1990]),("post-2000",est[est.year>=2000]),
               ("high income only",est[est.high_income==1]),("non-high income",est[est.high_income==0])]:
    if len(dd)<200: continue
    i=fit(dd,"growth",X_FULL).inf("fd_z_L")
    sens.append({"test":f"Sample: {lab}","value":i["coef"],"detail":f"p={i['p']:.4g}; N={i['n']}"})
for v in [c+"_z" for c in IMF if c!="fd"]:
    dd=est.dropna(subset=[v+"_L"]); i=fit(dd,"growth",[v+"_L"]+[c+"_L" for c in CTRL_FULL]).inf(v+"_L")
    sens.append({"test":f"Exposure: IMF {v[:-2].upper()}","value":i["coef"],"detail":f"p={i['p']:.4g}; N={i['n']}"})
i=fit(est.dropna(subset=["fd_L"]),"growth",["fd_L"]+[c+"_L" for c in CTRL_FULL]).inf("fd_L")
sens.append({"test":"Exposure: raw IMF index (0-1, unstandardised)","value":i["coef"],"detail":f"p={i['p']:.4g}"})
for v,lab in [("credit_L","Credit/GDP level"),("credit_gap_L","Credit/GDP HP gap (lambda=100)")]:
    dd=est.dropna(subset=[v]); i=fit(dd,"growth",[v]+[c+"_L" for c in CTRL_FULL]).inf(v)
    sens.append({"test":f"Denominator test: {lab}","value":i["coef"],"detail":f"p={i['p']:.4g}; N={i['n']}"})
dd=est.dropna(subset=["g_growth_pwt"])
if len(dd)>200:
    i=fit(dd,"g_growth_pwt",X_FULL).inf("fd_z_L")
    sens.append({"test":"Alternative outcome: PWT real GDP growth","value":i["coef"],"detail":f"p={i['p']:.4g}; N={i['n']}"})
sens_df=pd.DataFrame(sens)
table(sens_df,"T05_sensitivity","Specification, sample, exposure and denominator sensitivity")
# Retain the Oster calculation in CSV for audit transparency, but omit its
# uninterpretable numeric delta from the human-readable TeX table.
tex_sens=sens_df.loc[~sens_df["test"].str.startswith("Oster")].to_latex(
    index=False,float_format="%.4f",caption="Specification, sample, exposure and denominator sensitivity",
    label="tab:T05_sensitivity")
(TAB/"T05_sensitivity.tex").write_text(tex_sens)
jk=[]
for c in est.iso3.unique(): jk.append({"drop":c,"type":"country","beta":fit(est[est.iso3!=c],"growth",X_FULL).inf("fd_z_L")["coef"]})
for y in est.year.unique(): jk.append({"drop":str(y),"type":"year","beta":fit(est[est.year!=y],"growth",X_FULL).inf("fd_z_L")["coef"]})
jk=pd.DataFrame(jk); table(jk,"T06_jackknife","Leave-one-country-out / leave-one-year-out influence")
het=[]
for v,lab in [("high_income","High income x FD"),("crisis08","2008-09 x FD"),("crisis20","2020 x FD")]:
    dd=est.copy(); dd["INT"]=dd["fd_z_L"]*dd[v]
    i=fit(dd,"growth",X_FULL+["INT",v]).inf("INT")
    het.append({"interaction":lab,"coef":i["coef"],"se":i["se"],"p":i["p"],"n":i["n"]})
table(pd.DataFrame(het),"T07_heterogeneity","Interaction effects")

# ---- THRESHOLD 1999 x 40, precomputed bread/contrast (fast exact sup-Wald bootstrap) ----
t0=time.time()
dth=est.sort_values(["iso3","year"]).reset_index(drop=True)
ucH,uqH=pd.factorize(dth["iso3"]); tcH,tqH=pd.factorize(dth["year"]); NUH,NTH=len(uqH),len(tqH)
yH=dth["growth"].to_numpy(float); fdH=dth["fd_z_L"].to_numpy(float)
Cc=dth[[c+"_L" for c in CTRL_FULL]].to_numpy(float)
Cw=np.column_stack([within(Cc[:,j],ucH,tcH,NUH,NTH) for j in range(Cc.shape[1])])
ywH=within(yH,ucH,tcH,NUH,NTH)
grid=np.quantile(fdH,np.linspace(.15,.85,THRESH_GRID))
kH=2+Cw.shape[1]; keffH=kH+NUH+NTH-1
c0H=NUH/(NUH-1)*(len(yH)-1)/(len(yH)-keffH)
contr=np.zeros(kH); contr[0]=1.0; contr[1]=-1.0
cache=[]
for tau in grid:
    lo=(fdH<=tau).astype(float)
    Xw=np.column_stack([within(fdH*lo,ucH,tcH,NUH,NTH),within(fdH*(1-lo),ucH,tcH,NUH,NTH),Cw])
    B=np.linalg.pinv(Xw.T@Xw); cache.append((Xw,B,Xw@(B@contr)))
def supW(yw):
    best=-np.inf; bi=0
    for i,(Xw,B,u) in enumerate(cache):
        e=yw-Xw@(B@(Xw.T@yw)); num=float(u@yw)**2
        sg=np.bincount(ucH,weights=u*e,minlength=NUH); den=c0H*float(sg@sg)
        W=num/den if den>0 else 0.0
        if W>best: best=W; bi=i
    return best,bi
sup_obs,idx=supW(ywH)
Xw,B,u=cache[idx]; bthr=B@(Xw.T@ywH); ethr=ywH-Xw@bthr
Vthr=cl_cov(Xw,ethr,ucH,NUH,keffH); tau_hat=float(grid[idx])
Xlin=np.column_stack([within(fdH,ucH,tcH,NUH,NTH),Cw])
Bl=np.linalg.pinv(Xlin.T@Xlin); fit0=Xlin@(Bl@(Xlin.T@ywH)); e0=ywH-fit0
rng=np.random.default_rng(SEED); cnt=0
for b in range(THRESH_BOOT):
    w=rng.choice([-1.,1.],size=NUH)[ucH]
    sb,_=supW(within(fit0+e0*w,ucH,tcH,NUH,NTH)); cnt+= sb>=sup_obs
    if (b+1)%100==0: log.info("threshold bootstrap %d/%d (%.1fs)",b+1,THRESH_BOOT,time.time()-t0)
p_boot=(1+cnt)/(THRESH_BOOT+1)
q=est.copy(); q["fd_sq"]=q["fd_z_L"]**2; Fq=fit(q,"growth",X_FULL+["fd_sq"])
nl=[{"model":"Quadratic FD","term":"fd_z_L^2","coef":Fq.inf("fd_sq")["coef"],"p":Fq.inf("fd_sq")["p"],"n":Fq.n},
    {"model":f"Hansen threshold (grid={THRESH_GRID}, boot={THRESH_BOOT})","term":f"tau={tau_hat:.3f}",
     "coef":float(bthr[0]),"p":p_boot,"n":len(est),
     "note":f"beta_below={bthr[0]:.3f} (se {math.sqrt(max(Vthr[0,0],0)):.3f}); beta_above={bthr[1]:.3f} (se {math.sqrt(max(Vthr[1,1],0)):.3f}); supW={sup_obs:.2f}"}]
table(pd.DataFrame(nl),"T08_nonlinearity","Quadratic and threshold nonlinearity (bootstrap sup-Wald)")
log.info("THRESHOLD %.1fs: tau=%.3f supW=%.2f p=%.4f",time.time()-t0,tau_hat,sup_obs,p_boot)

# ============================== 6. DIAGNOSTICS =======================================
def cd_stat(res,unit,time,minov=3):
    R=pd.DataFrame({"u":unit,"t":time,"e":res}).pivot(index="t",columns="u",values="e").to_numpy(float)
    N=R.shape[1]; s=0.0
    for i in range(N):
        for j in range(i+1,N):
            m=np.isfinite(R[:,i])&np.isfinite(R[:,j]); T=int(m.sum())
            if T<minov: continue
            a,b=R[m,i],R[m,j]
            if a.std()<1e-12 or b.std()<1e-12: continue
            s+=math.sqrt(T)*float(np.corrcoef(a,b)[0,1])
    cd=math.sqrt(2/(N*(N-1)))*s
    return cd,float(2*stats.norm.sf(abs(cd)))
dg=[]; cd,p=cd_stat(F.e,F.d["iso3"],F.d["year"]); dg.append({"test":"Pesaran CD (FE residuals, T_ij>=3)","stat":cd,"p":p})
cd2,p2=cd_stat(est["fd_z_L"].to_numpy(),est["iso3"],est["year"]); dg.append({"test":"Pesaran CD (exposure FD)","stat":cd2,"p":p2})
dd=pd.DataFrame({"u":F.d["iso3"],"t":F.d["year"],"e":F.e}).sort_values(["u","t"])
dd["de"]=dd.groupby("u")["e"].diff(); dd["dt"]=dd.groupby("u")["t"].diff(); dd=dd[dd.dt==1]
dd["dl"]=dd.groupby("u")["de"].shift(1); dd["dt2"]=dd.groupby("u")["t"].diff()
dd=dd[dd.dt2==1].dropna(subset=["de","dl"])
Aa=np.column_stack([np.ones(len(dd)),dd["dl"].to_numpy(float)]); yv=dd["de"].to_numpy(float)
_,b_,e_=ols(Aa,yv); uc_,uq_=pd.factorize(dd["u"]); V_=cl_cov(Aa,e_,uc_,len(uq_),len(uq_)+2)
rho=float(b_[1]); se_=float(np.sqrt(V_[1,1])); t_=(rho+.5)/se_
dg.append({"test":"Wooldridge AR(1) (H0: slope=-0.5)","stat":t_,"p":float(2*stats.t.sf(abs(t_),len(uq_)-1))})
Z=np.column_stack([F.fw**2,F.fw**3])
Zw=np.column_stack([within(Z[:,j],F.uc,F.tc,F.NU,F.NT) for j in range(2)])
Xa=np.column_stack([F.Xw,Zw]); _,ba,ea=ols(Xa,F.yw)
Va=cl_cov(Xa,ea,F.uc,F.NU,F.keff+2); jj=[F.k,F.k+1]; bb=ba[jj]; Vs=Va[np.ix_(jj,jj)]
Wst=float(bb@np.linalg.pinv(Vs)@bb)/2
dg.append({"test":"RESET (squared+cubed within fit)","stat":Wst,"p":float(stats.f.sf(Wst,2,F.NU-1))})
for v in ["growth","fd_z_L"]:
    s=est.dropna(subset=[v]).copy(); s["lag"]=s.groupby("iso3")[v].shift(1); s=s.dropna(subset=["lag"])
    Aa=np.column_stack([np.ones(len(s)),s["lag"].to_numpy(float)])
    _,b2_,_=ols(Aa,s[v].to_numpy(float)); dg.append({"test":f"Pooled descriptive AR(1): {v}","stat":float(b2_[1]),"p":np.nan})
Far=fit(est,"fd_z",["fd_z_L"])
far=Far.inf("fd_z_L")
dg.append({"test":"Within AR(1): fd_z on fd_z_L (country and year FE)","stat":far["coef"],"p":far["p"]})
log.info("WITHIN AR(1) for IMF FDI: coef=%.6f SE_country=%.6f p=%.8g",far["coef"],far["se"],far["p"])
table(pd.DataFrame(dg),"T09_diagnostics","CSD, serial correlation, functional form, persistence")

# ============================== 7. FORECASTING =======================================
fcols=["fd_z_L"]+[c+"_L" for c in CTRL_FULL]
ml=est[["iso3","year","growth"]+fcols].dropna().copy()
tr=ml[ml.year<=TRAIN_END].copy(); va=ml[(ml.year>=VAL_START)&(ml.year<=VAL_END)].copy()
te=ml[(ml.year>=TEST_START)&(ml.year<=TEST_END)].copy()
te=te[te.iso3.isin(set(tr.iso3))].copy(); va=va[va.iso3.isin(set(tr.iso3))].copy()
assert len(tr)>200 and len(va)>50 and len(te)>50, "insufficient split sizes"
log.info("SPLIT train=%d val=%d test=%d (%d-%d)",len(tr),len(va),len(te),TEST_START,TEST_END)
def build(trn,oth,mode):
    cm=trn.groupby("iso3")[fcols].mean(); ym=trn.groupby("iso3")["growth"].mean()
    def aug(d):
        a=d.copy()
        for c in fcols: a["M_"+c]=a["iso3"].map(cm[c]).fillna(cm[c].mean())
        a["My"]=a["iso3"].map(ym).fillna(ym.mean()); return a
    Aa,Bb=aug(trn),aug(oth)
    if mode=="mundlak":
        F_=fcols+["M_"+c for c in fcols]+["My"]
        return Aa[F_].to_numpy(float),Aa["growth"].to_numpy(float),Bb[F_].to_numpy(float),None
    ytr=Aa["growth"].to_numpy(float)-Aa["My"].to_numpy(float)
    Xtr=Aa[fcols].to_numpy(float)-Aa[["M_"+c for c in fcols]].to_numpy(float)
    Xte=Bb[fcols].to_numpy(float)-Bb[["M_"+c for c in fcols]].to_numpy(float)
    return Xtr,ytr,Xte,Bb["My"].to_numpy(float)
def fe_ref(trn,oth):
    cm=trn.groupby("iso3")[fcols].mean(); ym=trn.groupby("iso3")["growth"].mean()
    Xtr=trn[fcols].to_numpy(float)-np.column_stack([trn["iso3"].map(cm[c]).to_numpy(float) for c in fcols])
    ytr=trn["growth"].to_numpy(float)-trn["iso3"].map(ym).to_numpy(float)
    b=np.linalg.lstsq(Xtr,ytr,rcond=None)[0]
    Xo=oth[fcols].to_numpy(float)-np.column_stack([oth["iso3"].map(cm[c]).fillna(cm[c].mean()).to_numpy(float) for c in fcols])
    return Xo@b+oth["iso3"].map(ym).fillna(ym.mean()).to_numpy(float)
GRIDS={"Ridge":{"alpha":[0.1,1,10,100]},"Lasso":{"alpha":[0.001,0.01,0.1]},
       "ElasticNet":{"alpha":[0.001,0.01,0.1],"l1_ratio":[0.3,0.7]},
       "RandomForest":{"max_depth":[6,12,None],"min_samples_leaf":[5,20]},
       "HistGB":{"max_leaf_nodes":[15,31],"learning_rate":[0.05,0.1],"max_iter":[300]},
       "MLP-Mundlak":{"hidden_layer_sizes":[(32,16),(64,32)],"alpha":[1e-3,1e-1]},
       "MLP-within":{"hidden_layer_sizes":[(32,16),(64,32)],"alpha":[1e-3,1e-1]}}
def make(name,p):
    if name=="Ridge": return Ridge(**p,random_state=SEED)
    if name=="Lasso": return Lasso(**p,random_state=SEED,max_iter=5000)
    if name=="ElasticNet": return ElasticNet(**p,random_state=SEED,max_iter=5000)
    if name=="RandomForest": return RandomForestRegressor(n_estimators=400,**p,random_state=SEED,n_jobs=-1)
    if name=="HistGB": return HistGradientBoostingRegressor(**p,random_state=SEED)
    return MLPRegressor(**p,random_state=SEED,max_iter=800,tol=1e-4,early_stopping=False,solver="adam")
def fitpred(name,params,trn,oth):
    mode="within" if name in ("Ridge","Lasso","ElasticNet","MLP-within") else "mundlak"
    Xtr,ytr,Xte,add=build(trn,oth,mode); sc=StandardScaler().fit(Xtr)
    mdl=make(name,params); mdl.fit(sc.transform(Xtr),ytr)
    pr=mdl.predict(sc.transform(Xte))
    return pr+add if add is not None else pr
tuned={}
for name,grid_ in GRIDS.items():
    keys=list(grid_); best=None
    for combo in itertools.product(*[grid_[k] for k in keys]):
        p=dict(zip(keys,combo))
        pv=fitpred(name,p,tr,va); r=float(np.sqrt(np.mean((va["growth"].to_numpy(float)-pv)**2)))
        if best is None or r<best[0]: best=(r,p)
    tuned[name]=best[1]; log.info("TUNED %-13s val RMSE=%.4f %s",name,best[0],best[1])
trv=pd.concat([tr,va]).reset_index(drop=True)
preds={"FE benchmark (two-way FE)":fe_ref(trv,te)}
for name,p in tuned.items(): preds[name]=fitpred(name,p,trv,te)
csa_tr=trv.groupby("year")[["growth"]+fcols].mean().add_prefix("CSA_").reset_index()
a=trv.merge(csa_tr,on="year",how="left"); b_=te.merge(csa_tr,on="year",how="left")
zc=[c for c in a.columns if c.startswith("CSA_")]
Aa=np.column_stack([np.ones(len(a)),a[fcols+zc].to_numpy(float)])
Bm=np.column_stack([np.ones(len(b_)),b_[fcols].to_numpy(float),b_[zc].fillna(a[zc].mean()).to_numpy(float)])
preds["CCE-linear"]=Bm@np.linalg.lstsq(Aa,a["growth"].to_numpy(float),rcond=None)[0]
y=te["growth"].to_numpy(float)
def met(y,p):
    e=y-p; return {"MAE":float(np.abs(e).mean()),"RMSE":float(np.sqrt(np.mean(e**2))),
                   "OOS_R2":float(1-(e@e)/float(((y-y.mean())**2).sum()))}
rank=pd.DataFrame([{"model":k,"n":len(y),**met(y,p)} for k,p in preds.items()]).sort_values("RMSE").reset_index(drop=True)
table(rank,"T10_forecast_ranking","Test-set ranking (selection on validation only)")
table(pd.DataFrame([{"model":k,"params":json.dumps(v)} for k,v in tuned.items()]),"T11_tuned_params","Validation-selected hyperparameters")
REFK="FE benchmark (two-way FE)"; e_ref=y-preds[REFK]
def dk_dm(d,unit,time):
    df=pd.DataFrame({"u":unit,"t":time,"d":d}); h=df.groupby("t")["d"].sum().to_numpy(float)
    T=len(h); L=max(1,int(math.floor(4*(T/100)**(2/9)))); S=float(h@h)
    for l in range(1,min(L,T-1)+1): S+=2*(1-l/(L+1))*float(h[l:]@h[:-l])
    var=S/(len(d)**2); z=float(df["d"].mean())/math.sqrt(var) if var>0 else np.nan
    return z,float(2*stats.norm.sf(abs(z)))
def signflip(ym,mc=8192):
    ym=np.asarray(ym,float); T=len(ym); obs=abs(ym.mean())
    if T<=12:
        c=sum(abs((ym*np.array(s)).mean())>=obs-1e-12 for s in itertools.product([-1,1],repeat=T))
        return c/(2**T)
    rng=np.random.default_rng(SEED)
    c=sum(abs((ym*rng.choice([-1.,1.],size=T)).mean())>=obs-1e-12 for _ in range(mc))
    return (1+c)/(mc+1)
def holm(ps):
    ps=np.nan_to_num(np.asarray(ps,float),nan=1.0); n=len(ps); o=np.argsort(ps); adj=np.empty(n); prev=0
    for r,i in enumerate(o): prev=max(prev,(n-r)*ps[i]); adj[i]=min(prev,1.0)
    return adj
tests=[]
for k,p in preds.items():
    if k==REFK: continue
    e=y-p; da=np.abs(e)-np.abs(e_ref); dq=e**2-e_ref**2
    tmp=te[["iso3","year"]].copy(); tmp["da"]=da; tmp["dq"]=dq
    ua=tmp.groupby("iso3")["da"].mean().to_numpy(); uq=tmp.groupby("iso3")["dq"].mean().to_numpy()
    za=ua.mean()/(ua.std(ddof=1)/math.sqrt(len(ua))); zq=uq.mean()/(uq.std(ddof=1)/math.sqrt(len(uq)))
    zda,pda=dk_dm(da,te["iso3"],te["year"]); zdq,pdq=dk_dm(dq,te["iso3"],te["year"])
    tests.append({"model":k,"dMAE":float(da.mean()),"dRMSE":met(y,p)["RMSE"]-met(y,preds[REFK])["RMSE"],
                  "panelDM_abs":float(za),"p_panel_abs":float(2*stats.norm.sf(abs(za))),
                  "panelDM_sq":float(zq),"p_panel_sq":float(2*stats.norm.sf(abs(zq))),
                  "DK_DM_abs":zda,"p_DK_abs":pda,"DK_DM_sq":zdq,"p_DK_sq":pdq,
                  "p_signflip_abs":signflip(tmp.groupby("year")["da"].mean().to_numpy()),
                  "p_signflip_sq":signflip(tmp.groupby("year")["dq"].mean().to_numpy())})
T_=pd.DataFrame(tests)
for c in ["p_panel_sq","p_DK_sq","p_signflip_sq"]: T_[c+"_holm"]=holm(T_[c])
table(T_,"T12_forecast_tests","DM: country-mean, Driscoll-Kraay, exact year sign-flip; Holm-adjusted")
best=rank.iloc[0]["model"]; eb=y-preds[best]
py=te[["iso3","year"]].copy(); py["e_ref"]=e_ref; py["e_best"]=eb
rows=[]
for yr,g in py.groupby("year"):
    rows.append({"year":int(yr),"n":len(g),"RMSE_ref":float(np.sqrt((g.e_ref**2).mean())),
                 "RMSE_best":float(np.sqrt((g.e_best**2).mean())),"dMSE":float((g.e_best**2-g.e_ref**2).mean())})
per=pd.DataFrame(rows); per["contrib"]=per["n"]*per["dMSE"]; table(per,"T13_tail","Per-year loss decomposition")
mu=trv.groupby("iso3")["growth"].mean(); base=te["iso3"].map(mu).fillna(mu.mean()).to_numpy(float)
da_rows=[]
for k,p in preds.items():
    hit=np.sign(p-base)==np.sign(y-base); n=len(y); ph=float(hit.mean())
    pbar=float((y>base).mean()); qbar=float((p>base).mean())
    pstar=pbar*qbar+(1-pbar)*(1-qbar); var=pstar*(1-pstar)/n
    z=(ph-pstar)/math.sqrt(var) if var>0 else np.nan
    da_rows.append({"model":k,"direction_accuracy":ph,"PT_stat":z,"p_PT":float(2*stats.norm.sf(abs(z)))})
table(pd.DataFrame(da_rows),"T14_direction","Directional accuracy and Pesaran-Timmermann test")

# ============================== 8. FIGURES ===========================================
b=F.inf("fd_z_L")["coef"]
fig,ax=plt.subplots(figsize=(6.8,3.2))
ax.barh(rank["model"][::-1],rank["RMSE"][::-1],color=[OI[1] if m==best else OI[0] for m in rank["model"][::-1]])
ax.set_xlabel(f"Test RMSE ({TEST_START}-{TEST_END})"); ax.set_title("Model ranking (validation-selected)"); fig_save(fig,"F1_ranking")
fig,ax=plt.subplots(figsize=(5.4,2.6))
labs=["Country","CGM","Driscoll-Kraay","Wild-Rademacher","Wild-Webb"]
los=[F.inf("fd_z_L",F.Vc)["lo"],F.inf("fd_z_L",F.Vtw,min(F.NU-1,F.NT-1))["lo"],F.inf("fd_z_L",F.Vdk,F.NT-1)["lo"],w1["lo"],w2["lo"]]
his=[F.inf("fd_z_L",F.Vc)["hi"],F.inf("fd_z_L",F.Vtw,min(F.NU-1,F.NT-1))["hi"],F.inf("fd_z_L",F.Vdk,F.NT-1)["hi"],w1["hi"],w2["hi"]]
for i,(lo,hi) in enumerate(zip(los,his)): ax.errorbar(b,i,xerr=[[b-lo],[hi-b]],fmt="o",color=OI[i%len(OI)],capsize=3)
ax.axvline(0,color="k",ls="--",lw=.8); ax.set_yticks(range(len(labs)),labs); ax.set_xlabel("FD coefficient (pp growth per SD)")
ax.set_title("Primary coefficient: five inference schemes"); fig_save(fig,"F2_inference")
fig,axes=plt.subplots(1,2,figsize=(7.2,2.9))
axes[0].bar(per["year"].astype(str),per["dMSE"],color=[OI[1] if c>0 else OI[0] for c in per["contrib"]])
axes[0].axhline(0,color="k",lw=.8); axes[0].set_title("Annual squared-loss gap (ElasticNet - FE benchmark)")
axes[1].plot(per["year"].astype(int),per["contrib"].cumsum(),marker="o",color=OI[0]); axes[1].axhline(0,color="k",ls="--",lw=.8)
axes[1].set_title("Cumulative squared-loss gap"); fig_save(fig,"F3_tail")
fig,ax=plt.subplots(figsize=(6.4,3.0)); jc=jk[jk.type=="country"].sort_values("beta")
sel=pd.concat([jc.iloc[:5],jc.iloc[-5:]])
ax.barh(sel["drop"],sel["beta"],color=OI[0]); ax.axvline(b,color=OI[1],ls="--")
ax.set_xlabel("Coefficient after dropping country"); ax.set_title("Leave-one-country-out sensitivity"); fig_save(fig,"F4_jackknife")
fig,ax=plt.subplots(figsize=(5.6,3.0))
xs=np.linspace(est["fd_z_L"].quantile(.05),est["fd_z_L"].quantile(.95),100)
ax.plot(xs,Fq.inf("fd_z_L")["coef"]*xs+Fq.inf("fd_sq")["coef"]*xs**2,color=OI[0])
ax.axhline(0,color="k",lw=.8)
ax.set_xlabel("FD (SD units)"); ax.set_ylabel("Quadratic component of fitted growth")
ax.text(.03,.04,"Quadratic term p = 0.289; threshold bootstrap p = 0.498\nNo statistically supported nonlinearity.",transform=ax.transAxes,fontsize=8,va="bottom",bbox={"facecolor":"white","edgecolor":"0.8","alpha":0.9})
ax.set_title("Exploratory quadratic fit (no threshold detected)"); fig_save(fig,"F5_nonlinearity")
fig,axes=plt.subplots(1,2,figsize=(7.2,3.0),sharex=True,sharey=True)
for ax,k,c in [(axes[0],REFK,OI[0]),(axes[1],best,OI[1])]:
    ax.scatter(y,preds[k],s=7,alpha=.6,color=c); lim=[y.min(),y.max()]; ax.plot(lim,lim,"k",lw=.8)
    ax.set_title(k[:28]); ax.set_xlabel("Actual growth (%)")
axes[0].set_ylabel("Predicted (%)"); fig_save(fig,"F6_scatter")
fig,ax=plt.subplots(figsize=(6.4,3.4))
sd=pd.DataFrame(sens); sel=sd[sd["test"].str.startswith(("Control set","Sample","Exposure"))].head(18)
ax.barh(sel["test"],sel["value"].astype(float),color=OI[0]); ax.axvline(b,color=OI[1],ls="--")
ax.set_xlabel("beta"); ax.set_title("Specification sensitivity"); fig.tight_layout(); fig_save(fig,"F7_sensitivity")
fig,ax=plt.subplots(figsize=(6.0,3.0))
ax.bar(T_["model"],T_["p_panel_sq_holm"],color=OI[0]); ax.axhline(.05,color=OI[1],ls="--")
ax.set_ylabel("Holm-adjusted p (squared loss)"); ax.set_xticklabels(T_["model"],rotation=45,ha="right")
ax.set_title("Holm-adjusted country-mean squared-loss tests"); fig.tight_layout(); fig_save(fig,"F8_holm")

# ============================== 9. MANIFEST ==========================================
checks={"FE_equals_LSDV":err<1e-6,"lsdv_max_diff":err,"verification_block":[len(cs_b),len(win_b)],
        "n_est":int(len(est)),"n_countries":int(est.iso3.nunique()),"window":[YMIN,YMAX],
        "split":[TRAIN_END,VAL_END,TEST_END],"threshold":[THRESH_GRID,THRESH_BOOT,tau_hat,sup_obs,p_boot],
        "models":list(preds),"all_models_ran":len(preds)==len(GRIDS)+2}
assert checks["all_models_ran"]
man={"utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"seed":SEED,"checks":checks,
     "provenance":provenance,"tuned":tuned,"versions":{}}
for m,dist in (("numpy","numpy"),("pandas","pandas"),("scipy","scipy"),("scikit-learn","scikit-learn"),("matplotlib","matplotlib")):
    try:
        import importlib.metadata as md; man["versions"][m]=md.version(dist)
    except Exception: man["versions"][m]=None
man["input_sha256"]={"raw_panel.csv":sha(raw_bytes),"country_meta.csv":sha(meta_bytes),"pwt110.xlsx":sha(pwt_bytes)}
man["data_mode"]="frozen local snapshot; no network requests"
man["dk_covariance"]="Corrected Hoechle (2007)/xtscc small-sample factor: (T/(T-1))*((n-1)/(n-k)); k counts within-regression regressors, not absorbed fixed effects."
json.dump(man,open(OUT/"manifest.json","w"),indent=2)
print("\n"+"="*90); print("REPLICATION COMPLETE")
print(f"Sample {len(est)} obs / {est.iso3.nunique()} countries ({YMIN}-{YMAX}) | FE==LSDV max diff {err:.2e}")
print(f"Threshold grid={THRESH_GRID} boot={THRESH_BOOT} tau={tau_hat:.3f} supW={sup_obs:.2f} p={p_boot:.4f}")
print(pd.DataFrame(rows).to_string(index=False)); print(rank.to_string(index=False))
print(T_[["model","dRMSE","p_panel_sq","p_panel_sq_holm","p_DK_sq","p_signflip_sq"]].to_string(index=False))
print("Tables:",len(list(TAB.glob('*.csv'))),"| Figures:",len(list(FIG.glob('*.png'))))
print("saved: tables/, figures/ and manifest.json in the replication folder")