"""Invariantes dos ajustes: restrições, episódios e alinhamento temporal."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Motor/notebooks'))
from consumo_melhorias import (choose_threshold, episode_metrics, aligned_forecasts,
    calibration_offset, choose_control, dynamics_frame)
from consumo_utils import RAW_SENSORS


def fake_data():
    parts=[]
    for e,offset in [('A',0),('B',100)]:
        x=pd.DataFrame({c:np.arange(12,dtype=float)+offset for c in RAW_SENSORS})
        x['Experimento']=e;x['Temp']=np.arange(12,dtype=float)
        x['Pi_6_log10']=0.;x['Pi_8']=.8;parts.append(x)
    return pd.concat(parts,ignore_index=True)


def test_threshold_minimizes_false_alerts_subject_to_recall():
    oof=pd.DataFrame({'Experimento':['A']*4,'truth':[1,1,0,0],'prob':[.9,.7,.6,.1]})
    value,feasible,table=choose_threshold(oof,[.5,.65,.8],.75)
    assert value==.65 and feasible


def test_episode_grouping_does_not_count_continuous_alarm_repeatedly():
    frame=pd.DataFrame({'Temp':np.arange(0,60,10),'truth':[0,1,1,0,0,0],
                        'decision':[1,1,1,0,1,1]})
    result=episode_metrics(frame)
    assert result['avisos']==2 and result['avisos_falsos']==1
    assert result['episodios_reais']==1 and result['episodios_detectados']==1
    assert result['avisos_falsos_min']==1
    assert result['antecedencia_confirmacao_mediana_s']==20
    assert result['antecedencia_decisao_correta_mediana_s']==10


def test_histories_are_compared_on_identical_origins():
    frames,features=aligned_forecasts(fake_data(),histories=(3,4),horizon=2,stride=2)
    pd.testing.assert_frame_equal(frames[3][['Experimento','Temp']],frames[4][['Experimento','Temp']])
    assert frames[3].query("Experimento=='A'").Temp.tolist()==[3,5,7,9]
    assert frames[3].query("Experimento=='A'").fuel_future_sum.tolist()==[9,13,17,21]


def test_calibration_learns_signed_residual_without_test_values():
    assert calibration_offset(np.array([4,6]),np.array([2,3]))==2.5


def test_controller_rejects_low_fuel_with_wrong_demand_or_support():
    # Velo, n, Tau, consumo: opção barata sem torque ou fora do suporte não serve.
    pred=np.array([[20,10,500,5],[20,10,100,1],[20,10,490,3]])
    chosen,accepted=choose_control(pred,[True,True,False],20,500,baseline=0)
    assert chosen==0 and accepted
    chosen,accepted=choose_control(pred,[False,False,False],20,500,baseline=0)
    assert chosen==0 and not accepted
    chosen,accepted=choose_control(pred,[True,True,True],20,500,baseline=0)
    assert chosen==2 and accepted


def test_dynamics_targets_are_next_second_and_stay_in_trial():
    frame,inputs,targets=dynamics_frame(fake_data())
    assert len(frame)==22
    a=frame.query("Experimento=='A' and Temp==10").iloc[0]
    assert a['next_Taccm']==11
    assert 'next_Taccm' not in inputs
    assert frame.groupby('Experimento').Temp.max().eq(10).all()
