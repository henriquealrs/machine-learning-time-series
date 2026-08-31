"""
===============================================================================
PIPELINE DE ANÁLISE DE TELEMETRIA VEICULAR
Engenharia de Recursos, Matrizes de Correlação, Métricas de Tráfego e Gráficos
===============================================================================
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
#%% 1. CONFIGURAÇÃO E ETL (CARGA + ENGENHARIA DE RECURSOS)
# =============================================================================

def carregar_e_preprocessar_dados(diretorio: str) -> dict[str, pd.DataFrame]:
    """Carrega os arquivos .txt do diretório e aplica a Engenharia de Recursos unificada."""
    arquivos = [
        f for f in os.listdir(diretorio) 
        if f.endswith(".txt") and not f.endswith("P.txt")
    ]
    dfs = {}
    relatorio_carga = []

    print(f"==+=+ Carregando e Processando Sessões ({len(arquivos)} arquivos) =+=+==\n")

    for arq in arquivos:
        caminho = os.path.join(diretorio, arq)
        df = pd.read_csv(caminho, sep="\t", decimal=",", encoding="latin-1")
        df = df.dropna(how="all", axis=1)
        df.columns = [c.strip() for c in df.columns]

        # 1. Padronização da Velocidade (km/h e m/s)
        if "TachographVehicleSpeed" in df.columns:
            df["Velocidade_kmh"] = df["TachographVehicleSpeed"]
        elif "GPS_Speed[m/s]" in df.columns:
            df["Velocidade_kmh"] = df["GPS_Speed[m/s]"] * 3.6
        else:
            df["Velocidade_kmh"] = 0.0

        vel_ms = df["Velocidade_kmh"] / 3.6

        # 2. Aceleração (m/s²) e Carga Aerodinâmica Proporcional (v²)
        df["Aceleracao"] = np.diff(vel_ms, prepend=vel_ms.iloc[0] if len(vel_ms) > 0 else 0)
        df["V_Quadrado"] = df["Velocidade_kmh"] ** 2

        # 3. Inclinação Filtrada da Via (%)
        if "GPS_Altitude[m]" in df.columns:
            # Janela de 15s para eliminação de ruído de satélite
            df["Alt_Suave_15s"] = df["GPS_Altitude[m]"].rolling(window=15, min_periods=1, center=True).mean()
            df["Delta_Alt"] = df["Alt_Suave_15s"].diff().fillna(0)
            df["Delta_Dist"] = vel_ms.fillna(0)  # m/s a 1Hz = metros percorridos em 1s

            df["Inclinacao_Filtrada[%]"] = np.where(
                df["Delta_Dist"] > 1.0,
                (df["Delta_Alt"] / df["Delta_Dist"]) * 100,
                0
            )
            df["Inclinacao_Filtrada[%]"] = df["Inclinacao_Filtrada[%]"].rolling(window=5, min_periods=1).mean()
        else:
            df["Inclinacao_Filtrada[%]"] = 0.0

        # 4. Densidade do Ar Estimada (kg/m³)
        if "AmbientAirTemp[C]" in df.columns:
            temp_k = df["AmbientAirTemp[C]"] + 273.15
            alt_m = df["GPS_Altitude[m]"] if "GPS_Altitude[m]" in df.columns else 0.0
            p_pascal = 101325 * (1 - 2.25577e-5 * alt_m) ** 5.25588
            df["Densidade_Ar[kg/m3]"] = p_pascal / (287.058 * temp_k)
        else:
            df["Densidade_Ar[kg/m3]"] = np.nan

        dfs[arq] = df
        relatorio_carga.append({
            "Arquivo": arq,
            "Dia": arq.split("_")[0],
            "Duração (s)": len(df),
            "Vel. Média (km/h)": df["Velocidade_kmh"].mean()
        })

    df_resumo = pd.DataFrame(relatorio_carga)
    print(df_resumo.to_string(index=False))
    print("\n" + "=" * 60 + "\n")
    return dfs


# =============================================================================
#%% 2. ANÁLISES ESTATÍSTICAS E MÉTRICAS DE TRÁFEGO
# =============================================================================

def analisar_correlacoes(dados_separados: dict[str, pd.DataFrame]):
    """Gera matrizes de correlação focadas no diagnóstico Motorista vs. Terreno."""
    print("==+=+ MATRIZES DE CORRELAÇÃO: MOTORISTA VS TERRENO +=+==\n")
    cols_diag = [
        "ActualEnginePercentTorque[percent]",
        "EngineFuelRate",
        "AcceleratorPedalAPPosition1",  # Intenção do Motorista
        "V_Quadrado",                   # Carga Aerodinâmica
        "Aceleracao",                   # Inércia
        "Inclinacao_Filtrada[%]"        # Terreno
    ]

    for nome, df in dados_separados.items():
        cols_val = [c for c in cols_diag if c in df.columns]
        if len(cols_val) > 1:
            print(f"=== Matriz de Diagnóstico: [{nome}] ===")
            matriz = df[cols_val].corr()
            cols_alvo = [c for c in ["ActualEnginePercentTorque[percent]", "EngineFuelRate"] if c in matriz.columns]
            print(matriz[cols_alvo].round(3))
            print("-" * 55)


def calcular_metricas_trafego(dados_separados: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calcula indicadores de tráfego, frenagens e instabilidade de condução."""
    resultados = []

    for nome, df in dados_separados.items():
        vel_kmh = df["Velocidade_kmh"]
        tempo_total_s = len(df)
        distancia_km = (vel_kmh / 3600.0).sum()

        # Detecção de eventos discretos de frenagem
        if "BrakePedalPosition" in df.columns:
            em_frenagem = df["BrakePedalPosition"] > 2.0
            eventos_frenagem = (em_frenagem & (~em_frenagem.shift(1, fill_value=False))).sum()
        else:
            em_frenagem = df["Aceleracao"] < -0.5
            eventos_frenagem = (em_frenagem & (~em_frenagem.shift(1, fill_value=False))).sum()

        frenagens_por_km = (eventos_frenagem / distancia_km) if distancia_km > 0 else np.nan

        # Métrica de paradas
        tempo_parada_s = (vel_kmh <= 1.0).sum()
        pct_tempo_parada = (tempo_parada_s / tempo_total_s) * 100.0 if tempo_total_s > 0 else 0

        # Variabilidade e instabilidade
        vel_media = vel_kmh.mean()
        vel_std = vel_kmh.std()
        coef_var = (vel_std / vel_media * 100.0) if vel_media > 0 else np.nan

        resultados.append({
            "Sessão": nome,
            "Distância (km)": round(distancia_km, 2),
            "Vel. Média (km/h)": round(vel_media, 1),
            "Vel. DesvPad (km/h)": round(vel_std, 2),
            "Coef. Variação (%)": round(coef_var, 1),
            "Tempo Parado (s)": tempo_parada_s,
            "% Tempo Parado": round(pct_tempo_parada, 1),
            "Nº Frenagens": eventos_frenagem,
            "Frenagens / km": round(frenagens_por_km, 2) if pd.notna(frenagens_por_km) else np.nan,
            "DesvPad Acel (m/s²)": round(df["Aceleracao"].std(), 3)
        })

    df_resumo = pd.DataFrame(resultados)
    print("\n" + "=" * 80)
    print("             DIAGNÓSTICO DE TRÁFEGO E DENSIDADE DE FRENAGENS")
    print("=" * 80)
    print(df_resumo.to_string(index=False))
    print("=" * 80 + "\n")
    return df_resumo


# =============================================================================
#%% 3. VISUALIZAÇÃO GRÁFICA
# =============================================================================

def gerar_graficos_diagnostico(dados_separados: dict[str, pd.DataFrame]):
    """Gera o conjunto completo de figuras de diagnóstico por sessão."""
    for nome, df in dados_separados.items():
        tempo_min = df.index / 60.0

        # --- FIGURA 1: Confronto Pedal vs Inclinação vs Resposta do Motor ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
        if "AcceleratorPedalAPPosition1" in df.columns:
            ax1.plot(tempo_min, df["AcceleratorPedalAPPosition1"], color="red", linewidth=1.5, label="Pedal Acelerador (%) [Motorista]")
        ax1.plot(tempo_min, df["Inclinacao_Filtrada[%]"], color="green", linewidth=2, linestyle="--", label="Inclinação (%) [Terreno]")
        ax1.set_ylabel("Entradas (Pedal / Terreno)")
        ax1.set_title(f"Confronto de Causalidade: Quem está exigindo o motor? ({nome})")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        if "ActualEnginePercentTorque[percent]" in df.columns:
            ax2.plot(tempo_min, df["ActualEnginePercentTorque[percent]"], color="blue", linewidth=1.5, label="Torque do Motor (%)")
        if "EngineFuelRate" in df.columns:
            ax2.plot(tempo_min, df["EngineFuelRate"], color="orange", alpha=0.7, label="Consumo (Fuel Rate)")
        ax2.set_ylabel("Resposta do Motor")
        ax2.set_xlabel("Tempo (min)")
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # --- FIGURA 2: Terreno + Temp. Óleo vs Marcha + Torque ---
        if "EngineOilTemperature[Celsius]" in df.columns:
            fig, (ax_top1, ax_bot1) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
            
            # Painel Superior
            l1 = ax_top1.plot(tempo_min, df["Inclinacao_Filtrada[%]"], color="tab:green", linewidth=1.8, label="Inclinação (%) [Terreno]")
            ax_top1.set_ylabel("Inclinação da Via (%)", color="tab:green")
            ax_top1.tick_params(axis="y", labelcolor="tab:green")
            ax_top1.grid(True, alpha=0.3)

            ax_top2 = ax_top1.twinx()
            l2 = ax_top2.plot(tempo_min, df["EngineOilTemperature[Celsius]"], color="tab:red", linewidth=2.0, label="Temp. Óleo (°C) [Resposta Térmica]")
            ax_top2.set_ylabel("Temp. Óleo (°C)", color="tab:red")
            ax_top2.tick_params(axis="y", labelcolor="tab:red")
            
            lines_top = l1 + l2
            ax_top1.legend(lines_top, [l.get_label() for l in lines_top], loc="upper left")
            ax_top1.set_title(f"Análise Causalidade: Terreno & Resp. Térmica vs. Marcha & Torque ({nome})")

            # Painel Inferior
            l3 = []
            if "CurrentGear" in df.columns:
                l3 = ax_bot1.plot(tempo_min, df["CurrentGear"], color="tab:purple", linewidth=1.5, drawstyle="steps-post", label="Marcha Engatada")
            ax_bot1.set_ylabel("Marcha (CurrentGear)", color="tab:purple")
            ax_bot1.tick_params(axis="y", labelcolor="tab:purple")
            ax_bot1.grid(True, alpha=0.3)

            ax_bot2 = ax_bot1.twinx()
            l4 = []
            if "ActualEnginePercentTorque[percent]" in df.columns:
                l4 = ax_bot2.plot(tempo_min, df["ActualEnginePercentTorque[percent]"], color="tab:blue", linewidth=1.2, alpha=0.5, label="Torque do Motor (%)")
            ax_bot2.set_ylabel("Torque do Motor (%)", color="tab:blue")
            ax_bot2.tick_params(axis="y", labelcolor="tab:blue")

            lines_bot = l3 + l4
            ax_bot1.legend(lines_bot, [l.get_label() for l in lines_bot], loc="upper left")
            ax_bot1.set_xlabel("Tempo (min)")
            plt.tight_layout()
            plt.show()

        # --- FIGURA 3: Condições Externas vs Dinâmica de Tráfego ---
        fig, (ax_top1, ax_bot1) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
        l1 = ax_top1.plot(tempo_min, df["Inclinacao_Filtrada[%]"], color="tab:green", linewidth=1.8, label="Inclinação (%) [Terreno]")
        ax_top1.set_ylabel("Inclinação da Via (%)", color="tab:green")
        ax_top1.tick_params(axis="y", labelcolor="tab:green")
        ax_top1.grid(True, alpha=0.3)

        ax_top2 = ax_top1.twinx()
        l2 = []
        if not df["Densidade_Ar[kg/m3]"].isna().all():
            l2 = ax_top2.plot(tempo_min, df["Densidade_Ar[kg/m3]"], color="tab:cyan", linewidth=2.0, linestyle="-.", label="Densidade do Ar (kg/m³)")
        ax_top2.set_ylabel("Densidade do Ar (kg/m³)", color="darkcyan")
        ax_top2.tick_params(axis="y", labelcolor="darkcyan")

        lines_top = l1 + l2
        ax_top1.legend(lines_top, [l.get_label() for l in lines_top], loc="upper left")
        ax_top1.set_title(f"Diagnóstico de Condições Externas & Dinâmica de Tráfego ({nome})")

        l3 = ax_bot1.plot(tempo_min, df["Aceleracao"], color="tab:blue", linewidth=1.2, alpha=0.6, label="Aceleração (m/s²)")
        ax_bot1.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax_bot1.set_ylabel("Aceleração (m/s²)", color="tab:blue")
        ax_bot1.tick_params(axis="y", labelcolor="tab:blue")
        ax_bot1.grid(True, alpha=0.3)

        ax_bot2 = ax_bot1.twinx()
        l4 = []
        if "BrakePedalPosition" in df.columns:
            l4 = ax_bot2.plot(tempo_min, df["BrakePedalPosition"], color="tab:red", linewidth=1.5, label="Pressão do Freio (%)")
        ax_bot2.set_ylabel("Pedal de Freio (%)", color="tab:red")
        ax_bot2.tick_params(axis="y", labelcolor="tab:red")

        lines_bot = l3 + l4
        ax_bot1.legend(lines_bot, [l.get_label() for l in lines_bot], loc="upper left")
        ax_bot1.set_xlabel("Tempo (min)")
        plt.tight_layout()
        plt.show()

        # --- FIGURA 4: Diagnóstico de Idling (Estado Parado) ---
        is_stopped = df["Velocidade_kmh"] <= 1.0
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

        # Destaque em vermelho transparente para estados de parada
        ax1.fill_between(tempo_min, 0, 1, where=is_stopped, color="red", alpha=0.18, transform=ax1.get_xaxis_transform(), label="Veículo Parado (v ≤ 1 km/h)")
        ax2.fill_between(tempo_min, 0, 1, where=is_stopped, color="red", alpha=0.18, transform=ax2.get_xaxis_transform(), label="Veículo Parado (v ≤ 1 km/h)")

        l1 = []
        if "EngineSpeed" in df.columns:
            l1 = ax1.plot(tempo_min, df["EngineSpeed"], color="tab:orange", linewidth=1.5, label="RPM do Motor")
        ax1.set_ylabel("RPM do Motor", color="tab:orange")
        ax1.tick_params(axis="y", labelcolor="tab:orange")
        ax1.grid(True, alpha=0.3)

        ax1_gear = ax1.twinx()
        l2 = []
        if "CurrentGear" in df.columns:
            l2 = ax1_gear.plot(tempo_min, df["CurrentGear"], color="tab:purple", linewidth=1.5, drawstyle="steps-post", label="Marcha Engatada")
        ax1_gear.set_ylabel("Marcha", color="tab:purple")
        ax1_gear.tick_params(axis="y", labelcolor="tab:purple")

        lines1 = l1 + l2
        ax1.legend(lines1, [l.get_label() for l in lines1], loc="upper left")
        ax1.set_title(f"Correlação de Paradas, Idling & Assinatura de Condução ({nome})")

        l3 = []
        if "BrakePedalPosition" in df.columns:
            l3 = ax2.plot(tempo_min, df["BrakePedalPosition"], color="tab:red", linewidth=1.5, label="Pressão do Freio (%)")
        ax2.set_ylabel("Pedal de Freio (%)", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax2.grid(True, alpha=0.3)

        ax2_acel = ax2.twinx()
        l4 = ax2_acel.plot(tempo_min, df["Aceleracao"], color="tab:blue", linewidth=1.2, alpha=0.6, label="Aceleração (m/s²)")
        ax2_acel.set_ylabel("Aceleração (m/s²)", color="tab:blue")
        ax2_acel.tick_params(axis="y", labelcolor="tab:blue")
        ax2_acel.axhline(0, color="gray", linestyle="--", alpha=0.5)

        lines2 = l3 + l4
        ax2.legend(lines2, [l.get_label() for l in lines2], loc="upper left")
        ax2.set_xlabel("Tempo (min)")
        plt.tight_layout()
        plt.show()


# =============================================================================
#%% 4. EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    # 1. Leitura de dados e feature engineering
    
    # Adicione o caminho dos dados em .txt
    CAMINHO_BASE = "./"
    dados = carregar_e_preprocessar_dados(CAMINHO_BASE)

    # 2. Resumos estatísticos e diagnósticos
    analisar_correlacoes(dados)
    df_trafego = calcular_metricas_trafego(dados)

    # 3. Exibição de gráficos (Descomente para visualizar)
    gerar_graficos_diagnostico(dados)
