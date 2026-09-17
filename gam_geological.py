"""
GAM SHAP Dependence - Geological Factors (GAM3)
- 5 features: Es, CQ, CCQ, FAO, QIK
- ALL SHAP=0 crossings + Bootstrap CI
- Legend: upper-left | R2: upper-right | S0: near x-axis
- Unified fonts, no bold axis labels, no overlap
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt, os, warnings
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pygam import LinearGAM, s
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
OUT_DIR = os.path.join(SCRIPT_DIR, "GAM3")
os.makedirs(OUT_DIR, exist_ok=True)

DATA_PATH = r"D:\岳的python机器学习3\1.0tust返稿\测试.xlsx"
SHAP_PATH = "TabPFN_SHAP_Analysis.xlsx"

GEO = ['Es\u200b\u200b', 'CQ', 'CCQ', 'FAO', 'QIK']
LABELS = {'Es\u200b\u200b': 'Es (MPa)', 'CQ': 'CQ (kPa)', 'CCQ': 'CCQ (kPa)',
          'FAO': 'FAO (kPa)', 'QIK': 'QIK (kPa)'}
PAL = {'Es\u200b\u200b': '#7b4ea3', 'CQ': '#2c3e50', 'CCQ': '#c9622d',
       'FAO': '#2e7193', 'QIK': '#3a8c5c'}

df = pd.read_excel(DATA_PATH).dropna()
X_all = df[GEO].values.astype(np.float64)
sidx = int(len(df)*0.8)
sy = StandardScaler(); sy.fit(df.iloc[:sidx,-1].values.reshape(-1,1)); y_std = sy.scale_[0]
shap = pd.read_excel(SHAP_PATH,sheet_name="SHAP_Values")
shap_r = shap[GEO].values * y_std
np.random.seed(42); ex = np.sort(np.random.choice(len(df)-sidx,27,replace=False))
X_ex = X_all[sidx:][ex]

plt.rcParams.update({
    'font.family':'serif','font.serif':['Times New Roman'],
    'font.size':14,'axes.labelsize':16,'xtick.labelsize':14,'ytick.labelsize':14,
    'legend.fontsize':13,'axes.unicode_minus':False,'axes.linewidth':1.6,
    'figure.dpi':400,'savefig.dpi':1200,
})

N_BOOT=500; BIN_N=8

def save_fig(fig,name):
    for ext in ['png','pdf']:
        fig.savefig(os.path.join(OUT_DIR,name+'.'+ext),dpi=1200,
                    bbox_inches='tight',facecolor='white',pad_inches=0.2)

# ===== GAM fitting =====
print("GAM3 -> "+OUT_DIR)
all_res=[]
for fi,feat in enumerate(GEO):
    x_raw=X_ex[:,fi]; y_shap=shap_r[:,fi]; c=PAL[feat]
    ql,qh=np.percentile(x_raw,[1,99]); m=(x_raw>=ql)&(x_raw<=qh); xp=x_raw[m]; yp_d=y_shap[m]; n_pts=len(xp)
    gam=LinearGAM(s(0,n_splines=20,spline_order=3,lam=3)).gridsearch(xp.reshape(-1,1),yp_d)
    bl=gam.lam; XX=gam.generate_X_grid(term=0,n=300); y_pred=gam.predict(XX)
    ci=gam.prediction_intervals(XX,width=0.95); xg,cl,cu=XX[:,0],ci[:,0],ci[:,1]
    r2=gam.statistics_['pseudo_r2']['explained_deviance']; pv=gam.statistics_['p_values'][0]
    edof=gam.statistics_['edof']

    crossings=[]
    for i in range(1,len(xg)):
        if y_pred[i-1]*y_pred[i]<0:
            t=xg[i-1]+(xg[i]-xg[i-1])*abs(y_pred[i-1])/(abs(y_pred[i-1])+abs(y_pred[i]))
            d='neg->pos' if y_pred[i-1]<0 else 'pos->neg'
            crossings.append({'x':t,'dir':d})

    n_orig=len(crossings)
    if n_orig>0:
        rng=np.random.RandomState(42); blists=[[] for _ in range(n_orig)]
        for _ in range(N_BOOT):
            ib=rng.choice(n_pts,n_pts,replace=True)
            try:
                gb=LinearGAM(s(0,n_splines=20,spline_order=3,lam=bl)).fit(xp[ib].reshape(-1,1),yp_d[ib])
                XXb=gb.generate_X_grid(term=0,n=200); ypb=gb.predict(XXb); xgb=XXb[:,0]
                bc=[];
                for j in range(1,len(xgb)):
                    if ypb[j-1]*ypb[j]<0: bc.append(xgb[j-1]+(xgb[j]-xgb[j-1])*abs(ypb[j-1])/(abs(ypb[j-1])+abs(ypb[j])))
                if len(bc)==n_orig:
                    for k in range(n_orig): blists[k].append(bc[k])
            except: pass
        for k,cr in enumerate(crossings):
            bv=blists[k]
            if len(bv)>=50: cr['lo'],cr['hi']=np.percentile(bv,[2.5,97.5])

    bc,bm,bs=[],[],[]
    if n_pts>=BIN_N*3:
        edges=np.percentile(xp,np.linspace(0,100,BIN_N+1))
        for k in range(BIN_N):
            m=(xp>=edges[k])&(xp<edges[k+1])
            if m.sum()>=2: bc.append(xp[m].mean()); bm.append(yp_d[m].mean()); bs.append(yp_d[m].std(ddof=1))

    all_res.append({'feat':feat,'label':LABELS[feat],'color':c,'r2':r2,'pv':pv,'edof':edof,
                    'xg':xg,'yp':y_pred,'cl':cl,'cu':cu,'xp':xp,'ys':yp_d,'crossings':crossings,
                    'bc':bc,'bm':bm,'bs':bs})
    ps='p<0.001' if pv<0.001 else ('p={:.3f}'.format(pv))
    print('  {}: R2={:.3f} {} edof={:.1f} crossings={}'.format(feat,r2,ps,edof,len(crossings)))
    for i,cr in enumerate(crossings):
        ci_s=''
        if 'lo' in cr: ci_s=' [{:.2f},{:.2f}]'.format(cr['lo'],cr['hi'])
        print('    S{}{}={:.2f} ({}){}'.format(chr(0x2080),chr(0x2080+i+1),cr['x'],cr['dir'],ci_s))

# ===== Plot function =====
def plot_geo(ax,res,is_panel=False):
    c=res['color']; xg,yp=res['xg'],res['yp']; cl,cu=res['cl'],res['cu']
    xp,ys=res['xp'],res['ys']; bc,bm,bs=res['bc'],res['bm'],res['bs']
    ax.set_facecolor('#fdfdfd')

    ax.fill_between(xg,0,yp,where=(yp>=0),color=c,alpha=0.06,zorder=0,lw=0)
    ax.fill_between(xg,0,yp,where=(yp<0),color=c,alpha=0.03,zorder=0,lw=0)
    ax.fill_between(xg,cl,cu,color=c,alpha=0.16,edgecolor='none',zorder=1)
    for cr in res['crossings']:
        if 'lo' in cr: ax.axvspan(cr['lo'],cr['hi'],alpha=0.14,color=c,lw=0,zorder=2)
    ax.axhline(0,color='#aaaaaa',lw=0.9,ls='-',alpha=0.45,zorder=3)
    if not is_panel: ax.scatter(xp,ys,color=c,s=18,alpha=0.30,edgecolors='none',zorder=4)
    if not is_panel and bc:
        ax.errorbar(bc,bm,yerr=bs,fmt='o',color='#444444',markersize=4.5,capsize=2.8,capthick=0.9,lw=0.9,alpha=0.55,zorder=5)
    ax.plot(xg,yp,'-',color=c,lw=3.0,zorder=6)
    for cr in res['crossings']: ax.axvline(cr['x'],color=c,ls='--',lw=1.8,alpha=0.75,zorder=7)

    # S0 annotations below curve (near x-axis, inside plot)
    if not is_panel and res['crossings']:
        n=len(res['crossings']); yl=ax.get_ylim(); yr=yl[1]-yl[0]
        if n==1: offs=[0.05]
        elif n==2: offs=[0.04,0.13]
        else: offs=list(np.linspace(0.03,0.14,n))
        for i,(cr,off) in enumerate(zip(res['crossings'],offs)):
            txt='S{}{}={:.2f}'.format(chr(0x2080),chr(0x2080+i+1),cr['x'])
            if 'lo' in cr: txt+='\n[{:.2f},{:.2f}]'.format(cr['lo'],cr['hi'])
            ax.text(cr['x'],yl[0]+off*yr,txt,fontsize=13,ha='center',va='bottom',color=c,
                    bbox=dict(boxstyle='round,pad=0.2',facecolor='white',edgecolor=c,alpha=0.93,lw=1.0))

    # R2 annotation (upper-right)
    ps='p<0.001' if res['pv']<0.001 else ('p<0.01' if res['pv']<0.01 else 'p={:.3f}'.format(res['pv']))
    stat='$R^2$ = {:.3f}   {}   edof = {:.1f}'.format(res['r2'],ps,res['edof'])
    ax.text(0.98,0.98,stat,transform=ax.transAxes,fontsize=13,ha='right',va='top',color='#000000',
            bbox=dict(boxstyle='round,pad=0.35',facecolor='white',edgecolor='#e0e0e0',alpha=0.90,lw=0.5))

    ax.set_xlabel(res['label'],fontsize=16,color='#333333')
    ax.set_ylabel('SHAP value (mm/min)',fontsize=16,color='#333333')
    ax.tick_params(which='minor',size=0)
    for sp in ax.spines.values(): sp.set_linewidth(1.8); sp.set_color('#cccccc')

# ===== Single figures =====
print('\nDrawing...')
for res in all_res:
    fig,ax=plt.subplots(figsize=(7.5,5.8))
    plot_geo(ax,res)
    leg=[Line2D([0],[0],color=res['color'],lw=3.0,label='GAM fit'),
         Patch(facecolor=res['color'],alpha=0.16,label='95% CI (fit)'),
         Line2D([0],[0],color=res['color'],lw=1.8,ls='--',label='SHAP=0 crossing'),
         Line2D([0],[0],marker='o',color='w',markerfacecolor='#444444',markersize=5,label='Binned mean +/- 1 SE')]
    ax.legend(handles=leg,loc='upper left',fontsize=13,framealpha=0.88,edgecolor='#dddddd',ncol=1)
    plt.tight_layout(pad=1.2)
    name='Fig_GAM_'+res['feat'].replace(chr(8203),'').strip()
    save_fig(fig,name); plt.close(fig)
    print('  + '+name)

# ===== Panel (lower DPI to avoid memory) =====
print('Drawing panel...')
fig_p,axes_p=plt.subplots(2,3,figsize=(19,12.5)); axes_p=axes_p.flatten()
for idx,res in enumerate(all_res): plot_geo(axes_p[idx],res,is_panel=True)
ax_leg=axes_p[5]; ax_leg.set_facecolor('#fdfdfd'); ax_leg.set_xlim(0,10); ax_leg.set_ylim(0,10); ax_leg.axis('off')
note=("GAM-based SHAP dependence analysis\nfor geological parameters\n\n"
      "Solid curve: GAM fit (n_splines=20)\nLight band: 95% CI of the GAM fit\n"
      "Dashed lines: SHAP=0 crossings\nError bars: binned SHAP mean +/- 1 SE\n"
      "Crossing CI: Bootstrap N="+str(N_BOOT)+"\n\n(Statistical association, not causation)")
ax_leg.text(5,5.5,note,transform=ax_leg.transAxes,fontsize=11,ha='center',va='center',color='#555555',
            bbox=dict(boxstyle='round,pad=0.8',facecolor='white',edgecolor='#dddddd',alpha=0.95,lw=1.0),
            linespacing=1.8,family='monospace')
plt.subplots_adjust(hspace=0.30,wspace=0.26,bottom=0.04,top=0.96)
fig_p.savefig(os.path.join(OUT_DIR,'Fig_GAM_Geological_Panel.png'),dpi=600,
              bbox_inches='tight',facecolor='white',pad_inches=0.15)
plt.close(fig_p)
print('  + Fig_GAM_Geological_Panel')

# Export
rows=[]
for res in all_res:
    r={'Feature':res['feat'],'R2':round(res['r2'],4),
       'p_value':'p<0.001' if res['pv']<0.001 else '{:.4f}'.format(res['pv']),
       'edof':round(res['edof'],1),'N_Crossings':len(res['crossings'])}
    for i,cr in enumerate(res['crossings']):
        r['S0_{}'.format(i+1)]=round(cr['x'],3); r['S0_{}_dir'.format(i+1)]=cr['dir']
        if 'lo' in cr: r['S0_{}_lo'.format(i+1)]=round(cr['lo'],3); r['S0_{}_hi'.format(i+1)]=round(cr['hi'],3)
    rows.append(r)
pd.DataFrame(rows).to_excel(os.path.join(OUT_DIR,'GAM_Geological_Results.xlsx'),index=False)
print('\nDone -> '+OUT_DIR)
