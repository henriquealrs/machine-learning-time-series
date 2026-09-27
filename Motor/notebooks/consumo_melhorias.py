"""Validação interna, calibração e simulação observacional de consumo.

Cada experimento preserva um ensaio externo sem participar da seleção.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import NearestNeighbors
from consumo_utils import forecast_frame, regressor


def classifier():
    return make_pipeline(SimpleImputer(strategy='median', add_indicator=True),
        RandomForestClassifier(n_estimators=200,max_depth=5,min_samples_leaf=8,
                               class_weight='balanced',random_state=42,n_jobs=1))


def choose_threshold(oof, thresholds=None, recall_min=.75):
    thresholds=np.round(np.arange(.30,.851,.05),2) if thresholds is None else thresholds
    rows=[]
    for threshold in thresholds:
        recalls=[];scores=[];fp=0
        for _,g in oof.groupby('Experimento'):
            pred=g.prob.to_numpy()>=threshold;y=g.truth.to_numpy()
            recalls.append(recall_score(y,pred,zero_division=0))
            scores.append(f1_score(y,pred,zero_division=0))
            fp+=int(((y==0)&pred).sum())
        rows.append(dict(limiar=float(threshold),recall_macro=np.mean(recalls),
                         f1_macro=np.mean(scores),fp_min=fp/(len(oof)*10/60)))
    table=pd.DataFrame(rows)
    feasible=table.recall_macro>=recall_min
    if feasible.any():
        best=table[feasible].sort_values(['fp_min','f1_macro','limiar'],ascending=[True,False,False]).iloc[0]
    else:
        best=table.sort_values(['recall_macro','fp_min'],ascending=[False,True]).iloc[0]
    return float(best.limiar),bool(feasible.any()),table


def episode_metrics(frame, step=10):
    """Um aviso por sequência positiva; tempo relativo à confirmação da janela."""
    g=frame.sort_values('Temp').reset_index(drop=True)
    truth=g.truth.to_numpy().astype(bool);pred=g.decision.to_numpy().astype(bool)
    times=g.Temp.to_numpy()
    def runs(mask):
        result=[];current=[]
        for i,active in enumerate(mask):
            if active:
                if current and times[i]-times[current[-1]]!=step:
                    result.append(current);current=[]
                current.append(i)
            elif current:
                result.append(current);current=[]
        if current:result.append(current)
        return result
    alarms=runs(pred);events=runs(truth)
    false=sum(not truth[idx].any() for idx in alarms)
    leads=[];correct_leads=[]
    for event in events:
        detected=[i for i in event if pred[i]]
        if detected:
            matching=[alarm for alarm in alarms if set(alarm).intersection(event)]
            notification=min(times[alarm[0]] for alarm in matching)
            leads.append(times[event[0]]+step-notification)
            correct_leads.append(times[event[0]]+step-times[detected[0]])
    duration=len(g)*step/60
    return dict(avisos=len(alarms),avisos_falsos=false,avisos_falsos_min=false/duration,
                episodios_reais=len(events),episodios_detectados=len(leads),
                recall_episodios=len(leads)/len(events) if events else np.nan,
                antecedencia_confirmacao_mediana_s=float(np.median(leads)) if leads else np.nan,
                antecedencia_decisao_correta_mediana_s=float(np.median(correct_leads)) if correct_leads else np.nan,
                decisoes_positivas=int(pred.sum()))


def context(frame):
    prefix='cur_' if 'cur_Acc' in frame else ''
    return np.select([frame[prefix+'Brake']>100,frame[prefix+'Velo']==0,
                      frame[prefix+'Brake']>0,
                      (frame[prefix+'Acc']>=60)&(frame[prefix+'Tau']>=600)],
                     ['freio_fora_faixa','parado','frenagem','demanda_alta'],default='movimento')


def run_alerts(data):
    frame,features=forecast_frame(data,horizon=10,history=10,stride=10)
    metrics=[];predictions=[];selections=[];tuning=[];provenance=[]
    for test_name in sorted(frame.Experimento.unique()):
        print('Alertas: teste',test_name,flush=True)
        train=frame[frame.Experimento!=test_name];test=frame[frame.Experimento==test_name]
        oof=[]
        for val_name in sorted(train.Experimento.unique()):
            fit=train[train.Experimento!=val_name];val=train[train.Experimento==val_name]
            cutoff=fit.fuel_future_mean.quantile(.75)
            model=classifier().fit(fit[features],fit.fuel_future_mean>cutoff)
            rows=val[['Experimento','Temp']].copy()
            rows['truth']=(val.fuel_future_mean>cutoff).astype(int)
            rows['prob']=model.predict_proba(val[features])[:,1]
            oof.append(rows)
            provenance.append(dict(teste_externo=test_name,validacao=val_name,
                                   treino=','.join(sorted(fit.Experimento.unique())),limite_U=cutoff))
        selected,feasible,scores=choose_threshold(pd.concat(oof,ignore_index=True))
        scores['teste_externo']=test_name;tuning.append(scores)
        cutoff=train.fuel_future_mean.quantile(.75)
        model=classifier().fit(train[features],train.fuel_future_mean>cutoff)
        probability=model.predict_proba(test[features])[:,1]
        y=(test.fuel_future_mean>cutoff).to_numpy()
        selections.append(dict(ensaio=test_name,limiar=selected,meta_viavel=feasible,
                               limite_U=cutoff,treino=','.join(sorted(train.Experimento.unique()))))
        for method,decision in [('persistencia',test.cur_Taccm.to_numpy()>cutoff),
                               ('limiar_050',probability>=.5),('limiar_ajustado',probability>=selected)]:
            rows=test[['Experimento','Temp','target_start','target_end','fuel_future_mean']].copy()
            rows['metodo']=method;rows['prob']=probability;rows['truth']=y;rows['decision']=decision
            rows['contexto']=context(test);rows['limiar']=selected if method=='limiar_ajustado' else (.5 if method=='limiar_050' else np.nan)
            predictions.append(rows)
            m=dict(ensaio=test_name,metodo=method,n=len(test),limiar=rows.limiar.iloc[0],
                precisao=precision_score(y,decision,zero_division=0),recall=recall_score(y,decision,zero_division=0),
                f1=f1_score(y,decision,zero_division=0),fp=int(((~y)&decision).sum()),
                fn=int((y&(~decision)).sum()))
            m['fp_min']=m['fp']/(len(test)*10/60);m.update(episode_metrics(rows))
            metrics.append(m)
    predictions=pd.concat(predictions,ignore_index=True)
    errors=[]
    for keys,g in predictions.groupby(['Experimento','metodo','contexto']):
        y=g.truth.astype(bool);p=g.decision.astype(bool)
        errors.append(dict(ensaio=keys[0],metodo=keys[1],contexto=keys[2],n=len(g),
            fp=int((~y&p).sum()),fn=int((y&~p).sum()),tp=int((y&p).sum())))
    return {'metricas':pd.DataFrame(metrics),'previsoes':predictions,
            'selecao':pd.DataFrame(selections),'busca_limiar':pd.concat(tuning,ignore_index=True),
            'contextos':pd.DataFrame(errors),'particoes_internas':pd.DataFrame(provenance)}


def aligned_forecasts(data,histories=(10,30),horizon=30,stride=30):
    frames={};feature_lists={};keys=['Experimento','Temp'];common=None
    for history in histories:
        f,features=forecast_frame(data,horizon=horizon,history=history,stride=1)
        frames[history]=f.set_index(keys);feature_lists[history]=features
        common=frames[history].index if common is None else common.intersection(frames[history].index)
    aligned=common.to_frame(index=False).sort_values(keys).reset_index(drop=True)
    aligned=aligned[aligned.groupby('Experimento').cumcount()%stride==0]
    index=pd.MultiIndex.from_frame(aligned[keys])
    return {h:f.loc[index].reset_index() for h,f in frames.items()},feature_lists


def calibration_offset(real,predicted):
    return float(np.mean(np.asarray(real)-np.asarray(predicted)))


def run_energy(data):
    frames,feature_lists=aligned_forecasts(data)
    metrics=[];predictions=[];selections=[];search=[];provenance=[]
    names=sorted(frames[10].Experimento.unique())
    for test_name in names:
        print('Energia: teste',test_name,flush=True)
        candidates=[];cache={}
        for history in (10,30):
            for pi6 in (True,False):
                f=frames[history];features=[c for c in feature_lists[history] if pi6 or c!='cur_Pi_6_log10']
                train=f[f.Experimento!=test_name];oof=[]
                for val_name in sorted(train.Experimento.unique()):
                    fit=train[train.Experimento!=val_name];val=train[train.Experimento==val_name]
                    model=regressor().fit(fit[features],fit.fuel_future_sum)
                    r=val[['Experimento','Temp','fuel_future_sum']].copy()
                    r['pred']=np.maximum(0,model.predict(val[features]));oof.append(r)
                    provenance.append(dict(teste_externo=test_name,historico=history,pi6=pi6,
                        validacao=val_name,treino=','.join(sorted(fit.Experimento.unique()))))
                oof=pd.concat(oof,ignore_index=True)
                fold_scores=[]
                for _,g in oof.groupby('Experimento'):
                    residual=g.pred-g.fuel_future_sum
                    fold_scores.append(residual.abs().mean()+abs(residual.mean()))
                candidate=dict(ensaio=test_name,historico=history,pi6=pi6,score_interno=np.mean(fold_scores),
                    correcao=calibration_offset(oof.fuel_future_sum,oof.pred))
                candidates.append(candidate);cache[(history,pi6)]=oof
        options=pd.DataFrame(candidates)
        best=options.sort_values(['score_interno','historico','pi6']).iloc[0]
        selections.append(best.to_dict());search.append(options)
        history=int(best.historico);pi6=bool(best.pi6);f=frames[history]
        features=[c for c in feature_lists[history] if pi6 or c!='cur_Pi_6_log10']
        train=f[f.Experimento!=test_name];test=f[f.Experimento==test_name]
        model=regressor().fit(train[features],train.fuel_future_sum)
        raw=np.maximum(0,model.predict(test[features]))
        forecasts={'persistencia':30*test.cur_Taccm.to_numpy(),
                   'mediana_treino':np.full(len(test),train.fuel_future_sum.median()),
                   'selecionado_sem_calibrar':raw,
                   'selecionado_calibrado':np.maximum(0,raw+best.correcao)}
        # A correção é sempre testada, não escolhida olhando o teste externo.
        for method,pred in forecasts.items():
            real=test.fuel_future_sum.to_numpy();residual=pred-real
            metrics.append(dict(ensaio=test_name,metodo=method,n=len(test),historico=history,pi6=pi6,
                mae=np.mean(np.abs(residual)),vies=np.mean(residual),total_real=real.sum(),
                total_previsto=pred.sum(),erro_total_pct=100*(pred.sum()/real.sum()-1)))
            rows=test[['Experimento','Temp','target_start','target_end','fuel_future_sum']].copy()
            rows['metodo']=method;rows['previsao']=pred;rows['residuo']=residual
            rows['contexto']=context(test);predictions.append(rows)
    predictions=pd.concat(predictions,ignore_index=True)
    regimes=predictions.groupby(['Experimento','metodo','contexto']).agg(
        n=('residuo','size'),vies=('residuo','mean'),mae=('residuo',lambda s:s.abs().mean())).reset_index()
    return {'metricas':pd.DataFrame(metrics),'previsoes':predictions,'selecao':pd.DataFrame(selections),
            'busca_configuracao':pd.concat(search,ignore_index=True),'contextos':regimes,
            'particoes_internas':pd.DataFrame(provenance)}


STATES=['Velo','n','Tau','Taccm']
DYN_INPUTS=STATES+['Acc','Brake','Gear','TeLam','Team']


def dynamics_frame(data):
    rows=[];targets=['next_'+c for c in STATES]
    for name,g in data.groupby('Experimento'):
        g=g.sort_values('Temp').reset_index(drop=True)
        if not g.Temp.diff().dropna().eq(1).all():raise ValueError('A dinâmica exige 1 segundo')
        f=g[['Experimento','Temp']+DYN_INPUTS].copy()
        for c in STATES:f['next_'+c]=g[c].shift(-1)
        rows.append(f.dropna(subset=targets))
    return pd.concat(rows,ignore_index=True),DYN_INPUTS,targets


def choose_control(predictions,supported,ref_v,ref_tau,baseline=1,vtol=2.,ttol=160.):
    p=np.asarray(predictions)
    valid=np.asarray(supported,dtype=bool)&(abs(p[:,0]-ref_v)<=vtol)&(abs(p[:,2]-ref_tau)<=ttol)
    if not valid.any():return baseline,False
    costs=np.where(valid,p[:,3],np.inf)
    # Empate mantém comando nominal, sem alteração gratuita.
    best=int(np.argmin(costs))
    if valid[baseline] and costs[baseline]==costs[best]:best=baseline
    return best,True


def run_control(data,horizon=10,vtol=2.,ttol=160.):
    frame,features,targets=dynamics_frame(data)
    one_step=[];rollouts=[];audits=[]
    for test_name in sorted(frame.Experimento.unique()):
        print('Dinâmica/simulação: teste',test_name,flush=True)
        train=frame[frame.Experimento!=test_name];test=frame[frame.Experimento==test_name]
        model=MultiOutputRegressor(regressor(),n_jobs=1).fit(train[features],train[targets])
        predicted=np.maximum(0,model.predict(test[features]))
        for i,c in enumerate(STATES):
            one_step.append(dict(ensaio=test_name,variavel=c,n=len(test),
                mae_modelo=mean_absolute_error(test[targets[i]],predicted[:,i]),
                mae_persistencia=mean_absolute_error(test[targets[i]],test[c])))
        scaler=make_pipeline(SimpleImputer(strategy='median'),StandardScaler())
        X=scaler.fit_transform(train[features])
        nn=NearestNeighbors(n_neighbors=2,algorithm='kd_tree').fit(X)
        limit=float(np.quantile(nn.kneighbors(X)[0][:,1],.95))
        audits.append(dict(ensaio=test_name,treino=','.join(sorted(train.Experimento.unique())),
                           limite_distancia=limit,tolerancia_velo=vtol,tolerancia_tau=ttol,
                           horizonte=horizon,delta_acc_max=5))
        g=data[data.Experimento==test_name].sort_values('Temp').reset_index(drop=True)
        starts=np.arange(0,len(g)-horizon,horizon)
        state_base=g.loc[starts,STATES].to_numpy().copy();state_policy=state_base.copy()
        for step in range(horizon):
            scheduled=g.loc[starts+step,features].reset_index(drop=True)
            reference=g.loc[starts+step+1,STATES].to_numpy()
            Xbase=scheduled.copy();Xbase[STATES]=state_base
            next_base=np.maximum(0,model.predict(Xbase))
            distance_base=nn.kneighbors(scaler.transform(Xbase),n_neighbors=1)[0][:,0]
            Xnominal=scheduled.copy();Xnominal[STATES]=state_policy
            repeated=Xnominal.iloc[np.repeat(np.arange(len(starts)),3)].reset_index(drop=True)
            acc=np.clip(scheduled.Acc.to_numpy()[:,None]+np.array([-5,0,5]),0,100)
            repeated['Acc']=acc.ravel()
            candidate=np.maximum(0,model.predict(repeated)).reshape(len(starts),3,4)
            distances=nn.kneighbors(scaler.transform(repeated),n_neighbors=1)[0].reshape(len(starts),3)
            distance_nominal=nn.kneighbors(scaler.transform(Xnominal),n_neighbors=1)[0][:,0]
            support=(distances<=limit)&(distance_nominal[:,None]<=limit)
            decisions=[choose_control(candidate[i],support[i],reference[i,0],reference[i,2],
                                      vtol=vtol,ttol=ttol) for i in range(len(starts))]
            choice=np.array([d[0] for d in decisions]);accepted=np.array([d[1] for d in decisions])
            next_policy=candidate[np.arange(len(starts)),choice]
            policy_acc=acc[np.arange(len(starts)),choice]
            for method,pred,command in [('replay',next_base,scheduled.Acc.to_numpy()),
                                        ('controlador_simulado',next_policy,policy_acc)]:
                result=pd.DataFrame({'Experimento':test_name,'janela':starts,'passo':step+1,
                    'Temp':g.loc[starts+step,'Temp'].to_numpy(),'metodo':method,
                    'Acc':command,'Acc_nominal':scheduled.Acc.to_numpy(),
                    'aceito':accepted if method=='controlador_simulado' else False,
                    'alterado':(~np.isclose(command,scheduled.Acc.to_numpy())) if method=='controlador_simulado' else False,
                    'distancia_nominal':distance_base if method=='replay' else distance_nominal,'limite_distancia':limit})
                for j,c in enumerate(STATES):result[c+'_previsto']=pred[:,j];result[c+'_referencia']=reference[:,j]
                result['violacao_velo']=abs(pred[:,0]-reference[:,0])>vtol
                result['violacao_tau']=abs(pred[:,2]-reference[:,2])>ttol
                rollouts.append(result)
            state_base=next_base;state_policy=next_policy
    rollout=pd.concat(rollouts,ignore_index=True)
    summaries=[]
    for (experiment,method),g in rollout.groupby(['Experimento','metodo']):
        sums=g.groupby('janela')[['Taccm_previsto','Taccm_referencia']].sum()
        summaries.append(dict(ensaio=experiment,metodo=method,janelas=len(sums),passos=len(g),
            integral_simulada=g.Taccm_previsto.sum(),integral_observada_referencia=g.Taccm_referencia.sum(),
            mae_integral_replay=mean_absolute_error(sums.Taccm_referencia,sums.Taccm_previsto) if method=='replay' else np.nan,
            mae_velo=mean_absolute_error(g.Velo_referencia,g.Velo_previsto),
            mae_tau=mean_absolute_error(g.Tau_referencia,g.Tau_previsto),
            fracao_violacao_velo=g.violacao_velo.mean(),fracao_violacao_tau=g.violacao_tau.mean(),
            fracao_aceita=g.aceito.mean(),fracao_alterada=g.alterado.mean(),
            fracao_fallback=1-g.aceito.mean() if method=='controlador_simulado' else np.nan))
    return {'dinamica_1s':pd.DataFrame(one_step),'simulacao_passos':rollout,
            'simulacao_resumo':pd.DataFrame(summaries),'parametros':pd.DataFrame(audits)}
