# -*- coding: utf-8 -*-

"""
============================================================
PLOTS DOS GRUPOS BUCKINGHAM-PI
============================================================

Lê:

    Dataset_Dimensional_Final.csv
    Dataset_Expressoes_Pi.csv

e gera:

    Pi_1.png
    Pi_2.png
    ...
    Pi_13.png

Cada gráfico contém os quatro ensaios:

    D1T1A
    D1T1B
    D1T2
    D2T1

Eixo X:
    Temp

Eixo Y:
    Pi_j

Título:
    Pi_j
    Expressão do Pi
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
# caminho_alvo = (os.path.normpath( os.path.join(pasta_atual, "Dados") )+"\\" ).replace("\\", "/")

# diretorio = ( caminho_alvo
    
# )

pasta_atual = os.getcwd()
caminho_alvo = (os.path.normpath( os.path.join(pasta_atual, "Dados") )+"\\" ).replace("\\", "/")

diretorio = ( caminho_alvo
    
)

arquivo_data = "Dataset_Dimensional_Final.csv"
arquivo_expressoes = "Dataset_Expressoes_Pi.csv"

caminho_data = os.path.join(
    diretorio,
    arquivo_data
)

caminho_expressoes = os.path.join(
    diretorio,
    arquivo_expressoes
)


# ============================================================
# 2. LEITURA DO DATASET DIMENSIONAL
# ============================================================

print("=" * 70)
print("LEITURA DO DATASET DIMENSIONAL")
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
# 3. LEITURA DAS EXPRESSÕES
# ============================================================

print("\n" + "=" * 70)
print("LEITURA DAS EXPRESSÕES DOS PI")
print("=" * 70)

expressoes = pd.read_csv(
    caminho_expressoes,
    sep=";",
    encoding="latin-1"
)

# Limpar espaços dos nomes das colunas
expressoes.columns = (
    expressoes.columns
    .astype(str)
    .str.strip()
)

# Limpar conteúdo
expressoes["PI"] = (
    expressoes["PI"]
    .astype(str)
    .str.strip()
)

expressoes["Expressão"] = (
    expressoes["Expressão"]
    .astype(str)
    .str.strip()
)


print(expressoes.to_string(index=False))


# ============================================================
# 4. CRIAR MAPA PI -> EXPRESSÃO
# ============================================================

Pi_expressions = dict(
    zip(
        expressoes["PI"],
        expressoes["Expressão"]
    )
)


# ============================================================
# 5. ENSAIOS
# ============================================================

experimentos = [
    "D1T1A",
    "D1T1B",
    "D1T2",
    "D2T1"
]


# ============================================================
# 6. GRUPOS PI
# ============================================================

Pi_names = [
    coluna
    for coluna in data.columns
    if coluna.startswith("Pi_")
    and coluna != "N_Pi_validos"
]

# Ordenar corretamente:
# Pi_1, Pi_2, ..., Pi_13

Pi_names = sorted(
    Pi_names,
    key=lambda x: int(
        x.split("_")[1]
    )
)


print("\n" + "=" * 70)
print("GRUPOS ENCONTRADOS")
print("=" * 70)

for pi in Pi_names:

    expressao = Pi_expressions.get(
        pi,
        "Expressão não encontrada"
    )

    print(
        f"{pi:7s} -> {expressao}"
    )


# ============================================================
# 7. VERIFICAR SE TODAS AS EXPRESSÕES EXISTEM
# ============================================================

expressoes_faltantes = [
    pi
    for pi in Pi_names
    if pi not in Pi_expressions
]

if expressoes_faltantes:

    print("\n" + "=" * 70)
    print("ATENÇÃO - EXPRESSÕES AUSENTES")
    print("=" * 70)

    for pi in expressoes_faltantes:
        print(f"  {pi}")


# ============================================================
# 8. DIRETÓRIO DOS GRÁFICOS
# ============================================================

diretorio_graficos = os.path.join(
    diretorio,
    "Graficos_Pi"
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
print("GERANDO GRÁFICOS")
print("=" * 70)

for pi in Pi_names:

    # --------------------------------------------------------
    # Expressão
    # --------------------------------------------------------

    expressao = Pi_expressions.get(
        pi,
        "Expressão não encontrada"
    )

    # --------------------------------------------------------
    # Figura
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    total_validos = 0

    # --------------------------------------------------------
    # Plotar cada experimento
    # --------------------------------------------------------

    for experimento in experimentos:

        subset = data[
            data["Experimento"] == experimento
        ].copy()

        # Converter para número
        subset["Temp"] = pd.to_numeric(
            subset["Temp"],
            errors="coerce"
        )

        subset[pi] = pd.to_numeric(
            subset[pi],
            errors="coerce"
        )

        # ----------------------------------------------------
        # Somente pontos com X e Y válidos
        # ----------------------------------------------------

        subset = subset[
            subset["Temp"].notna()
            &
            subset[pi].notna()
        ]

        n_validos = len(subset)

        total_validos += n_validos

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        if n_validos > 0:

            ax.scatter(
                subset["Temp"],
                subset[pi],
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

    # Título em duas linhas
    ax.set_title(
        f"{pi}\n{expressao}",
        fontsize=12
    )

    ax.set_xlabel(
        "Tempo da amostra [s]"
    )

    ax.set_ylabel(
        pi
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
        f"{pi}.png"
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

    # Fecha a figura para não acumular
    # plt.close(fig)

    print(
        f"{pi:7s} -> "
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
    f"\nGráficos salvos em:"
    f"\n{diretorio_graficos}"
)

print(
    f"\nTotal de gráficos = "
    f"{len(Pi_names)}"
)