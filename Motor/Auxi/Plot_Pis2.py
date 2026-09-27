# -*- coding: utf-8 -*-

"""
============================================================
PROGRAMA 4 - PLOTS DAS TRANSFORMAÇÕES PARA MODELAGEM
============================================================

Lê:

    Dataset_Modelagem_Pi.csv

E gera gráficos das variáveis transformadas:

    Pi_5_log10
    Pi_6_log10
    Pi_7_rel
    Pi_7_pct
    Pi_8_pct
    Pi_9_log10
    Pi_13_log10

Cada gráfico contém os quatro ensaios:

    D1T1A
    D1T1B
    D2T1
    D2T2

Eixo X:
    Temp

Eixo Y:
    variável transformada

As figuras permanecem abertas no Spyder para inspeção visual.

Os arquivos PNG são salvos em:

    Graficos_Modelagem_Pi

============================================================
"""


# ============================================================
# IMPORTAÇÕES
# ============================================================

import os

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. ARQUIVOS
# ============================================================

# pasta_atual = os.getcwd()
# caminho_alvo = (os.path.normpath( os.path.join(pasta_atual, "..", "Dados") )+"\\" ).replace("\\", "/")

# diretorio = ( caminho_alvo
    
# )

pasta_atual = os.getcwd()
caminho_alvo = (os.path.normpath( os.path.join(pasta_atual, "Dados") )+"\\" ).replace("\\", "/")

diretorio = ( caminho_alvo
    
)

arquivo_data = "Dataset_Modelagem_Pi.csv"

caminho_data = os.path.join(
    diretorio,
    arquivo_data
)


# ============================================================
# 2. LEITURA DO DATASET DE MODELAGEM
# ============================================================

print("=" * 70)
print("LEITURA DO DATASET DE MODELAGEM")
print("=" * 70)

data = pd.read_csv(
    caminho_data,
    sep=";",
    decimal=",",
    encoding="latin-1"
)

print(
    f"Linhas  = {data.shape[0]}"
)

print(
    f"Colunas = {data.shape[1]}"
)


# ============================================================
# 3. ENSAIOS
# ============================================================

experimentos = [
    "D1T1A",
    "D1T1B",
    "D1T2",
    "D2T1"
]


# ============================================================
# 4. VARIÁVEIS DE MODELAGEM A PLOTAR
# ============================================================

variaveis_modelagem = [
    "Pi_5_log10",
    "Pi_6_log10",
    "Pi_7_rel",
    "Pi_7_pct",
    "Pi_8_pct",
    "Pi_9_log10",
    "Pi_13_log10"
]


# ============================================================
# 5. TÍTULOS / EXPRESSÕES
# ============================================================

titulos = {

    "Pi_5_log10":
        "log10(Pi_5)\n"
        "Pi_5 = Diem / (Temp² · Velo³)",

    "Pi_6_log10":
        "log10(Pi_6)\n"
        "Pi_6 = Taccm / (Temp² · Velo³)",

    "Pi_7_rel":
        "Pi_7_rel = Pi_7 - 1\n"
        "Pi_7 = Teom / TeLam",

    "Pi_7_pct":
        "Pi_7_pct = 100 · (Pi_7 - 1)\n"
        "Pi_7 = Teom / TeLam",

    "Pi_8_pct":
        "Pi_8_pct = 100 · Pi_8\n"
        "Pi_8 = Team / TeLam",

    "Pi_9_log10":
        "log10(Pi_9)\n"
        "Pi_9 = x / (Temp · Velo)",

    "Pi_13_log10":
        "log10(Pi_13)\n"
        "Pi_13 = Velo² · mass / Tau"
}


# ============================================================
# 6. VERIFICAR VARIÁVEIS EXISTENTES
# ============================================================

print("\n" + "=" * 70)
print("VARIÁVEIS DE MODELAGEM")
print("=" * 70)

variaveis_faltantes = []

for variavel in variaveis_modelagem:

    if variavel in data.columns:

        print(
            f"{variavel:15s} -> OK"
        )

    else:

        print(
            f"{variavel:15s} -> NÃO ENCONTRADA"
        )

        variaveis_faltantes.append(
            variavel
        )


# ============================================================
# 7. PARAR CASO ALGUMA VARIÁVEL NÃO EXISTA
# ============================================================

if variaveis_faltantes:

    print("\n" + "=" * 70)
    print("ERRO - VARIÁVEIS AUSENTES")
    print("=" * 70)

    for variavel in variaveis_faltantes:

        print(
            f"  {variavel}"
        )

    raise ValueError(
        "Existem variáveis de modelagem ausentes no dataset."
    )


# ============================================================
# 8. DIRETÓRIO DOS GRÁFICOS
# ============================================================

diretorio_graficos = os.path.join(
    diretorio,
    "Graficos_Modelagem_Pi"
)

os.makedirs(
    diretorio_graficos,
    exist_ok=True
)


# ============================================================
# 9. MARCADORES DOS ENSAIOS
# ============================================================

markers = {

    "D1T1A": "o",

    "D1T1B": "s",

    "D1T2": "^",

    "D2T1": "D"
}


# ============================================================
# 10. GERAR OS GRÁFICOS
# ============================================================

print("\n" + "=" * 70)
print("GERANDO GRÁFICOS DAS TRANSFORMAÇÕES")
print("=" * 70)


for variavel in variaveis_modelagem:


    # ========================================================
    # TÍTULO
    # ========================================================

    titulo = titulos.get(
        variavel,
        variavel
    )


    # ========================================================
    # FIGURA
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )


    total_validos = 0


    # ========================================================
    # PLOTAR CADA EXPERIMENTO
    # ========================================================

    for experimento in experimentos:


        subset = data[
            data["Experimento"] == experimento
        ].copy()


        # ----------------------------------------------------
        # Converter para número
        # ----------------------------------------------------

        subset["Temp"] = pd.to_numeric(
            subset["Temp"],
            errors="coerce"
        )

        subset[variavel] = pd.to_numeric(
            subset[variavel],
            errors="coerce"
        )


        # ----------------------------------------------------
        # Somente pontos válidos
        # ----------------------------------------------------

        subset = subset[
            subset["Temp"].notna()
            &
            subset[variavel].notna()
        ]


        n_validos = len(
            subset
        )

        total_validos += n_validos


        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        if n_validos > 0:

            ax.scatter(

                subset["Temp"],

                subset[variavel],

                s=18,

                alpha=0.65,

                marker=markers[experimento],

                label=(
                    f"{experimento} "
                    f"(n={n_validos})"
                )
            )


    # ========================================================
    # FORMATAÇÃO
    # ========================================================

    ax.set_title(
        titulo,
        fontsize=12
    )

    ax.set_xlabel(
        "Tempo da amostra [s]"
    )

    ax.set_ylabel(
        variavel
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()


    fig.tight_layout()


    # ========================================================
    # SALVAR
    # ========================================================

    nome_arquivo = (
        f"{variavel}.png"
    )

    caminho_saida = os.path.join(
        diretorio_graficos,
        nome_arquivo
    )


    fig.savefig(
        caminho_saida,
        dpi=300,
        bbox_inches="tight"
    )


    # ========================================================
    # NÃO FECHAR
    # ========================================================

    # plt.close(fig)


    print(
        f"{variavel:15s} -> "
        f"{total_validos} pontos válidos -> "
        f"{caminho_saida}"
    )


# ============================================================
# 11. FINAL
# ============================================================

print("\n" + "=" * 70)
print("CONCLUÍDO")
print("=" * 70)

print(
    "\nGráficos salvos em:"
    f"\n{diretorio_graficos}"
)

print(
    "\nTotal de gráficos = "
    f"{len(variaveis_modelagem)}"
)

print(
    "\nAs figuras permanecem abertas no Spyder."
)
