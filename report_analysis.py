import json, os, numpy as np, pandas as pd, train_models as em
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import wilcoxon, spearmanr
import process_emg; em.RUTA_DATASET=os.environ.get('EMG_RAW', process_emg.RUTA_DATASET)
os.makedirs('rep', exist_ok=True)
OUT={}
d=em.cargar_datos(); E=em.escenarios(d); D=E['E1'].reset_index(drop=True); y=D.etiqueta.values.astype(int)
def per_subj(pred,thr):
    rows=[]
    for s,g in pred.groupby('Sujeto'):
        p=(g.s>=thr).astype(int)
        rows.append(dict(acc=accuracy_score(g.y,p),prec=precision_score(g.y,p,average='macro',zero_division=0),
             rec=recall_score(g.y,p,average='macro',zero_division=0),f1=f1_score(g.y,p,average='macro',zero_division=0),
             auc=roc_auc_score(g.y,g.s) if g.y.nunique()==2 and g.s.nunique()>1 else np.nan))
    r=pd.DataFrame(rows); return {k:(float(r[k].mean()),float(r[k].std())) for k in r}
def contr(pred): return pred.groupby(['Sujeto','contr_id']).agg(y=('y','first'),s=('s','mean')).reset_index()
def loso_lr(feats,Dx=D,mask_tr=None):
    yy=Dx.etiqueta.values.astype(int); s=np.zeros(len(Dx))
    for tr,te in LeaveOneGroupOut().split(Dx,yy,Dx.Sujeto):
        m=em.make_pipeline(em.StandardScaler(),em.LogisticRegression(C=1.0,max_iter=2000)).fit(Dx.iloc[tr][feats].values,yy[tr])
        s[te]=m.predict_proba(Dx.iloc[te][feats].values)[:,1]
    return pd.DataFrame({'Sujeto':Dx.Sujeto.values,'contr_id':Dx.contr_id.values,'y':yy,'s':s})
# A. main table
preds={}
s=np.zeros(len(D))
for tr,te in LeaveOneGroupOut().split(D,y,D.Sujeto): s[te]=np.bincount(y[tr]).argmax()
preds['Majority class']=(pd.DataFrame({'Sujeto':D.Sujeto,'contr_id':D.contr_id,'y':y,'s':s}),0.5)
preds['Logistic regression (RMS + MDF)']=(loso_lr(['rms_n','mdf_n']),0.5)
nombres={'Regresión logística':'Logistic regression (9 features)','LDA':'LDA (9 features)','Naive Bayes':'Gaussian naive Bayes','KNN (k=25)':'KNN (k = 25)','Random Forest':'Random forest','Gradient Boosting':'Gradient boosting (HGB)','SVM lineal':'Linear SVM','SVM RBF':'RBF SVM','MLP (64-32)':'MLP (64-32)'}
for k,v in nombres.items():
    p=pd.read_csv(f"Modelos/predicciones/E1_{''.join(ch for ch in k if ch.isalnum())}.csv"); preds[v]=(p,0.0 if k.startswith('SVM') else 0.5)
sc=np.zeros(len(D))
for suj in D.Sujeto.unique(): sc[(D.Sujeto==suj).values]=np.load(f'Modelos/predicciones/cnn_E1/{suj}.npy')
preds['1D CNN (raw signal)']=(pd.DataFrame({'Sujeto':D.Sujeto,'contr_id':D.contr_id,'y':y,'s':sc}),0.5)
OUT['main_window']={k:per_subj(p,t) for k,(p,t) in preds.items()}
OUT['main_contr']={k:per_subj(contr(p),t) for k,(p,t) in preds.items()}
# B per class best
p,t=preds['Logistic regression (9 features)']
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
for lvl,P in [('window',p),('contr',contr(p))]:
    yp=(P.s>=t).astype(int); pr,rc,f1,sup=precision_recall_fscore_support(P.y,yp,labels=[0,1])
    OUT['perclass_'+lvl]=dict(prec=pr.tolist(),rec=rc.tolist(),f1=f1.tolist(),sup=sup.tolist(),acc=float(accuracy_score(P.y,yp)),cm=confusion_matrix(P.y,yp).tolist())
# per subject best (window & contr macro F1)
ps=[]
pc=contr(p)
for suj,g in p.groupby('Sujeto'):
    gc=pc[pc.Sujeto==suj]
    ps.append(dict(Sujeto=suj,f1_w=f1_score(g.y,g.s>=.5,average='macro'),f1_c=f1_score(gc.y,gc.s>=.5,average='macro'),acc_w=accuracy_score(g.y,g.s>=.5),acc_c=accuracy_score(gc.y,gc.s>=.5),n_contr=len(gc)))
PS=pd.DataFrame(ps); PS.to_csv('rep/per_subject.csv',index=False)
# D failure analysis
S=pd.read_csv('QC/qc_sujetos.csv'); A=pd.read_csv('QC/qc_archivos.csv')
leg=S[S.apto_entrenamiento][['Sujeto','Pierna','mdf_final_vs_inicio_serie','rms_final_vs_inicio_serie','snr_mediana_db','ipi_mediana_s']]
subj=leg.groupby('Sujeto').mean(numeric_only=True).reset_index().merge(PS,on='Sujeto')
OUT['fail_corr']={c:[float(x) for x in spearmanr(subj[c],subj.acc_c)] for c in ['mdf_final_vs_inicio_serie','rms_final_vs_inicio_serie','snr_mediana_db','ipi_mediana_s']}
OUT['subj_table']=subj.round(3).to_dict('records')
# errors by position & set
pc2=pc.merge(D.groupby('contr_id').agg(pos=('pos_en_serie','first'),Serie=('Serie','first'),Pierna=('Pierna','first')).reset_index(),on='contr_id')
pc2['err']=((pc2.s>=.5).astype(int)!=pc2.y)
pc2['zona']=np.where(pc2.y==0,np.where(pc2.pos<=0.125,'NF-early (0-12.5%)','NF-late (12.5-25%)'),np.where(pc2.pos>=0.875,'F-late (87.5-100%)','F-early (75-87.5%)'))
OUT['err_zone']=pc2.groupby('zona').err.agg(['mean','count']).round(3).reset_index().to_dict('records')
OUT['err_set']=pc2.groupby('Serie').err.agg(['mean','count']).round(3).reset_index().to_dict('records')
OUT['err_leg']=pc2.groupby('Pierna').err.agg(['mean','count']).round(3).reset_index().to_dict('records')
OUT['err_class']=pc2.groupby('y').err.mean().round(3).to_dict()
# misclassified fatigue contractions feature summary
win=D.copy(); win['s']=p.s.values
cf=win.groupby('contr_id').agg(y=('etiqueta','first'),s=('s','mean'),mdf=('mdf_n','median'),rms=('rms_n','median'))
cf['ok']=(cf.s>=.5)==(cf.y==1)
OUT['mis_feat']=cf.groupby(['y','ok'])[['mdf','rms']].median().apply(np.exp).round(3).reset_index().to_dict('records')
# C bilateral
both=leg.groupby('Sujeto').filter(lambda g: len(g)==2)
pv=both.pivot(index='Sujeto',columns='Pierna',values=['mdf_final_vs_inicio_serie','rms_final_vs_inicio_serie'])
OUT['bilat_n']=int(len(pv))
for v in ['mdf_final_vs_inicio_serie','rms_final_vs_inicio_serie']:
    r=pv[v]['Derecha']; l=pv[v]['Izquierda']
    OUT['bilat_'+v]=dict(right=(float(r.median()),float(r.quantile(.25)),float(r.quantile(.75))),left=(float(l.median()),float(l.quantile(.25)),float(l.quantile(.75))),p=float(wilcoxon(r,l).pvalue),rho=float(spearmanr(r,l)[0]))
# set duration and contraction count decline per leg (all usable files)
A_ok=A[A.estado=='OK'].copy(); A_ok['Serie']=pd.to_numeric(A_ok.Serie,errors='coerce')
dec=[]
for (s_,l_),g in A_ok.groupby(['Sujeto','Pierna']):
    g=g.sort_values('Serie')
    if len(g)>=3 and (g.Serie==1).any(): dec.append(dict(Sujeto=s_,Pierna=l_,ratio_contr=g.contr_detectadas.iloc[-1]/g.contr_detectadas.iloc[0],n1=g.contr_detectadas.iloc[0]))
DE=pd.DataFrame(dec); dp=DE.pivot(index='Sujeto',columns='Pierna',values='ratio_contr').dropna()
OUT['bilat_contrratio']=dict(n=int(len(dp)),right=float(dp.Derecha.median()),left=float(dp.Izquierda.median()),p=float(wilcoxon(dp.Derecha,dp.Izquierda).pvalue))
# cross-leg transfer
def cross(train_leg,test_leg):
    s=[];rows=[]
    out=[]
    for suj in D.Sujeto.unique():
        tr=(D.Sujeto!=suj)&(D.Pierna==train_leg); te=(D.Sujeto==suj)&(D.Pierna==test_leg)
        if te.sum()==0: continue
        m=em.make_pipeline(em.StandardScaler(),em.LogisticRegression(max_iter=2000)).fit(D.loc[tr,em.CARACT],D.loc[tr,'etiqueta'].astype(int))
        out.append(pd.DataFrame({'Sujeto':suj,'contr_id':D.loc[te,'contr_id'],'y':D.loc[te,'etiqueta'].astype(int),'s':m.predict_proba(D.loc[te,em.CARACT])[:,1]}))
    P=pd.concat(out); return per_subj(contr(P),.5)['acc'], per_subj(P,.5)['acc'], P.Sujeto.nunique()
OUT['transfer']={f'{a}->{b}':cross(a,b) for a in ['Derecha','Izquierda'] for b in ['Derecha','Izquierda']}
# E label shift sensitivity
dd=d[(d.estado_archivo=='OK')&(d.apto_entrenamiento==True)&d[em.CARACT].notna().all(axis=1)].copy()
sens={}
for q in [0.20,0.25,0.33,0.40]:
    X=dd.copy(); X['etiqueta']=np.where(X.pos_en_serie<=q,0,np.where(X.pos_en_serie>=1-q,1,np.nan)); X=X[X.etiqueta.notna()].reset_index(drop=True)
    P=loso_lr(em.CARACT,X); sens[q]=dict(contr=per_subj(contr(P),.5)['acc'],win=per_subj(P,.5)['acc'],n=int(len(X)))
OUT['label_shift']=sens
# also kappa-like agreement between q=0.25 labels and "shifted by one contraction"
# F ablation
groups={'All 9 features':em.CARACT,'Without amplitude (RMS, MAV, WL)':['zc_n','ssc_n','mnf_n','mdf_n','fi_nsm5_n','ratio_bajo_alto_n'],
 'Without spectral (MNF, MDF, FInsm5, ratio)':['rms_n','mav_n','wl_n','zc_n','ssc_n'],'Without ZC and SSC':['rms_n','mav_n','wl_n','mnf_n','mdf_n','fi_nsm5_n','ratio_bajo_alto_n'],
 'Spectral only':['mnf_n','mdf_n','fi_nsm5_n','ratio_bajo_alto_n'],'Amplitude only':['rms_n','mav_n','wl_n'],'RMS + MDF':['rms_n','mdf_n']}
OUT['ablation']={k:dict(win=per_subj(loso_lr(v),.5)['acc'],contr=per_subj(contr(loso_lr(v)),.5)['acc']) for k,v in groups.items()}
# H table 2.2
A2=A.copy(); A2['Serie']=pd.to_numeric(A2.Serie,errors='coerce')
OUT['contr_per_set']={l:(float(g.contr_detectadas.mean()),float(g.contr_detectadas.std()),int(g.contr_detectadas.min()),int(g.contr_detectadas.max()),int(len(g))) for l,g in A2[A2.estado=='OK'].groupby('Pierna')}
OUT['dur_per_set']={l:(float(g.duracion_s.mean()),float(g.duracion_s.std()),float(g.duracion_s.min()),float(g.duracion_s.max())) for l,g in A2[A2.estado=='OK'].groupby('Pierna')}
# I windows withheld
OUT['win_withheld']=int(dd.etiqueta.isna().sum()); OUT['win_total_E1units']=int(len(dd)); OUT['contr_withheld']=int(dd[dd.etiqueta.isna()].contr_id.nunique())
# J versions
import sys, sklearn, scipy, matplotlib, numpy, pandas
OUT['versions']=dict(python=sys.version.split()[0],numpy=numpy.__version__,pandas=pandas.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,matplotlib=matplotlib.__version__)
json.dump(OUT,open('rep/resultados_reporte.json','w'),indent=1,default=float)
print(json.dumps({k:OUT[k] for k in ['perclass_window','perclass_contr','fail_corr','err_zone','err_set','err_leg','err_class','mis_feat','bilat_n','bilat_mdf_final_vs_inicio_serie','bilat_rms_final_vs_inicio_serie','bilat_contrratio','transfer','label_shift','ablation','contr_per_set','dur_per_set','win_withheld','win_total_E1units','contr_withheld','versions']},indent=0,default=float))