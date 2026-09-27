# -*- coding: utf-8 -*-

"""
============================================================
TRATAMENTO ESTATÍSTICO DOS GRUPOS BUCKINGHAM-PI
============================================================

Entrada:

    Dataset_Dimensional_Final.csv

Saída:

    Dataset_Modelagem_Pi.csv
    Dataset_Transformacoes_Pi.csv

IMPORTANTE:

    O Dataset_Dimensional_Final.csv NÃO é alterado.

As transformações são aplicadas somente em novas colunas.

============================================================
"""

# ============================================================
# IMPORTAÇÕES
# ============================================================

import os
import numpy as np
import pandas as pd


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

arquivo_entrada = "Dataset_Dimensional_Final.csv"

arquivo_saida = "Dataset_Modelagem_Pi.csv"

arquivo_transformacoes = (
    "Dataset_Transformacoes_Pi.csv"
)


caminho_entrada = os.path.join(
    diretorio,
    arquivo_entrada
)

caminho_saida = os.path.join(
    diretorio,
    arquivo_saida
)

caminho_transformacoes = os.path.join(
    diretorio,
    arquivo_transformacoes
)


# ============================================================
# 2. LEITURA
# ============================================================

print("=" * 70)
print("LEITURA DO DATASET DIMENSIONAL")
print("=" * 70)

data = pd.read_csv(
    caminho_entrada,
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
# 3. COPIAR DATASET
# ============================================================

model_data = data.copy()


# ============================================================
# 4. FUNÇÃO PARA LOG10
# ============================================================

def log10_positive(series):
    """
    Calcula log10 somente para valores estritamente positivos.

    Valores:
        > 0  -> log10(valor)
        <= 0 -> NaN
        NaN  -> NaN
    """

    values = pd.to_numeric(
        series,
        errors="coerce"
    )

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype=float
    )

    valid = (
        values.notna()
        &
        np.isfinite(values)
        &
        (values > 0)
    )

    result.loc[valid] = np.log10(
        values.loc[valid]
    )

    return result


# ============================================================
# 5. Pi_1
# ============================================================

# Acc
#
# Mantemos o valor original.
#
# Não criamos transformação.

print("\nPi_1 -> Acc")
print("Tratamento: original")


# ============================================================
# 6. Pi_2
# ============================================================

# Brake
#
# Mantemos o valor original.
#
# O valor 102 será apenas diagnosticado.

print("\nPi_2 -> Brake")
print("Tratamento: original")


# ============================================================
# 7. Pi_3
# ============================================================

# Gear
#
# Mantemos o valor original.
#
# É uma variável discreta:
# 1, 2, 3, 4

print("\nPi_3 -> Gear")
print("Tratamento: original / discreta")


# ============================================================
# 8. Pi_4
# ============================================================

# Pi_4 = Temp * n
#
# Por enquanto mantemos somente o original.
#
# Queremos preservar a física e investigar posteriormente
# a forte dependência temporal.

print("\nPi_4 -> Temp * n")
print("Tratamento: original")


# ============================================================
# 9. Pi_5
# ============================================================

# Pi_5 = Diem / (Temp^2 * Velo^3)
#
# Criamos:
#
#     Pi_5_log10 = log10(Pi_5)
#
# Somente valores > 0 entram no log.

if "Pi_5" in model_data.columns:

    model_data["Pi_5_log10"] = (
        log10_positive(
            model_data["Pi_5"]
        )
    )

    print("\nPi_5 -> log10 criado")


# ============================================================
# 10. Pi_6
# ============================================================

# Pi_6 = Taccm / (Temp^2 * Velo^3)
#
# Criamos:
#
#     Pi_6_log10

if "Pi_6" in model_data.columns:

    model_data["Pi_6_log10"] = (
        log10_positive(
            model_data["Pi_6"]
        )
    )

    print("\nPi_6 -> log10 criado")


# ============================================================
# 11. Pi_7
# ============================================================

# Pi_7 = Teom / TeLam
#
# Criamos:
#
#     Pi_7_rel = Pi_7 - 1
#
# Isso representa o desvio relativo da razão em relação a 1.
#
# Também criamos:
#
#     Pi_7_pct = 100 * (Pi_7 - 1)

if "Pi_7" in model_data.columns:

    model_data["Pi_7_rel"] = (
        model_data["Pi_7"] - 1.0
    )

    model_data["Pi_7_pct"] = (
        100.0
        * model_data["Pi_7_rel"]
    )

    print(
        "\nPi_7 -> "
        "Pi_7_rel e Pi_7_pct criados"
    )


# ============================================================
# 12. Pi_8
# ============================================================

# Pi_8 = Team / TeLam
#
# Mantemos:
#
#     Pi_8
#
# e criamos somente uma versão percentual:
#
#     Pi_8_pct = 100 * Pi_8
#
# Isso é uma mudança de escala para apresentação/interpretação,
# não uma nova grandeza física.

if "Pi_8" in model_data.columns:

    model_data["Pi_8_pct"] = (
        100.0
        * model_data["Pi_8"]
    )

    print(
        "\nPi_8 -> Pi_8_pct criado"
    )


# ============================================================
# 13. Pi_9
# ============================================================

# Pi_9 = x / (Temp * Velo)
#
# Criamos:
#
#     Pi_9_log10
#
# Somente valores > 0.

if "Pi_9" in model_data.columns:

    model_data["Pi_9_log10"] = (
        log10_positive(
            model_data["Pi_9"]
        )
    )

    print("\nPi_9 -> log10 criado")


# ============================================================
# 14. Pi_10
# ============================================================

# Pi_10 = zAl / (Temp * Velo)
#
# Por enquanto mantemos somente o original.
#
# O outlier será investigado antes de qualquer transformação.

print("\nPi_10 -> original")
print("Tratamento: investigar outlier")


# ============================================================
# 15. Pi_11
# ============================================================

# Pi_11 pode ser positivo ou negativo.
#
# Log comum não é apropriado.
#
# Mantemos original.

print("\nPi_11 -> original")


# ============================================================
# 16. Pi_12
# ============================================================

# Pi_12 pode ser positivo ou negativo.
#
# Log comum não é apropriado.
#
# Mantemos original.

print("\nPi_12 -> original")


# ============================================================
# 17. Pi_13
# ============================================================

# Pi_13 = Velo^2 * mass / Tau
#
# Temos valores iguais a zero.
#
# Para o log:
#
#     log10(0)
#
# não existe.
#
# Portanto:
#
#     Pi_13 > 0 -> log10(Pi_13)
#     Pi_13 <= 0 -> NaN
#
# NÃO utilizamos 1E-20.

if "Pi_13" in model_data.columns:

    model_data["Pi_13_log10"] = (
        log10_positive(
            model_data["Pi_13"]
        )
    )

    print("\nPi_13 -> log10 criado")


# ============================================================
# 18. DIAGNÓSTICO DAS TRANSFORMAÇÕES
# ============================================================

print("\n" + "=" * 70)
print("DIAGNÓSTICO DAS TRANSFORMAÇÕES")
print("=" * 70)

transformation_rows = []


# ------------------------------------------------------------
# Pi_5
# ------------------------------------------------------------

if "Pi_5" in model_data.columns:

    original = model_data["Pi_5"]

    valid_original = (
        original.notna()
    )

    valid_log = (
        model_data["Pi_5_log10"].notna()
    )

    transformation_rows.append({
        "Pi": "Pi_5",
        "Transformacao": "log10",
        "N_original": int(
            valid_original.sum()
        ),
        "N_transformado": int(
            valid_log.sum()
        ),
        "N_na_transformacao":
            int(
                valid_original.sum()
                - valid_log.sum()
            )
    })


# ------------------------------------------------------------
# Pi_6
# ------------------------------------------------------------

if "Pi_6" in model_data.columns:

    original = model_data["Pi_6"]

    valid_original = (
        original.notna()
    )

    valid_log = (
        model_data["Pi_6_log10"].notna()
    )

    transformation_rows.append({
        "Pi": "Pi_6",
        "Transformacao": "log10",
        "N_original": int(
            valid_original.sum()
        ),
        "N_transformado": int(
            valid_log.sum()
        ),
        "N_na_transformacao":
            int(
                valid_original.sum()
                - valid_log.sum()
            )
    })


# ------------------------------------------------------------
# Pi_7
# ------------------------------------------------------------

if "Pi_7" in model_data.columns:

    transformation_rows.append({
        "Pi": "Pi_7",
        "Transformacao": "Pi_7 - 1",
        "N_original": int(
            model_data["Pi_7"].notna().sum()
        ),
        "N_transformado": int(
            model_data["Pi_7_rel"].notna().sum()
        ),
        "N_na_transformacao": 0
    })


# ------------------------------------------------------------
# Pi_8
# ------------------------------------------------------------

if "Pi_8" in model_data.columns:

    transformation_rows.append({
        "Pi": "Pi_8",
        "Transformacao": "100 * Pi_8",
        "N_original": int(
            model_data["Pi_8"].notna().sum()
        ),
        "N_transformado": int(
            model_data["Pi_8_pct"].notna().sum()
        ),
        "N_na_transformacao": 0
    })


# ------------------------------------------------------------
# Pi_9
# ------------------------------------------------------------

if "Pi_9" in model_data.columns:

    original = model_data["Pi_9"]

    valid_original = (
        original.notna()
    )

    valid_log = (
        model_data["Pi_9_log10"].notna()
    )

    transformation_rows.append({
        "Pi": "Pi_9",
        "Transformacao": "log10",
        "N_original": int(
            valid_original.sum()
        ),
        "N_transformado": int(
            valid_log.sum()
        ),
        "N_na_transformacao":
            int(
                valid_original.sum()
                - valid_log.sum()
            )
    })


# ------------------------------------------------------------
# Pi_13
# ------------------------------------------------------------

if "Pi_13" in model_data.columns:

    original = model_data["Pi_13"]

    valid_original = (
        original.notna()
    )

    valid_log = (
        model_data["Pi_13_log10"].notna()
    )

    transformation_rows.append({
        "Pi": "Pi_13",
        "Transformacao": "log10",
        "N_original": int(
            valid_original.sum()
        ),
        "N_transformado": int(
            valid_log.sum()
        ),
        "N_na_transformacao":
            int(
                valid_original.sum()
                - valid_log.sum()
            )
    })


transformations = pd.DataFrame(
    transformation_rows
)


print(
    transformations.to_string(
        index=False
    )
)


# ============================================================
# 19. DIAGNÓSTICO ESPECIAL - BRAKE
# ============================================================

print("\n" + "=" * 70)
print("DIAGNÓSTICO Pi_2 / BRAKE")
print("=" * 70)

if "Pi_2" in model_data.columns:

    brake = model_data["Pi_2"]

    print(
        f"Máximo = "
        f"{brake.max()}"
    )

    print(
        f"Valores > 100 = "
        f"{(brake > 100).sum()}"
    )

    if (brake > 100).any():

        print(
            "\nExistem valores de Brake "
            "acima de 100."
        )

        print(
            "Eles foram preservados."
        )


# ============================================================
# 20. VERIFICAR NOVAS COLUNAS
# ============================================================

print("\n" + "=" * 70)
print("NOVAS COLUNAS")
print("=" * 70)

novas_colunas = [
    coluna
    for coluna in model_data.columns
    if coluna not in data.columns
]

for coluna in novas_colunas:

    print(
        f"  {coluna}"
    )


# ============================================================
# 21. SALVAR DATASET DE MODELAGEM
# ============================================================

model_data.to_csv(
    caminho_saida,
    sep=";",
    decimal=",",
    encoding="latin-1",
    index=False
)


# ============================================================
# 22. SALVAR TABELA DE TRANSFORMAÇÕES
# ============================================================

transformations.to_csv(
    caminho_transformacoes,
    sep=";",
    decimal=",",
    encoding="latin-1",
    index=False
)


# ============================================================
# 23. FINAL
# ============================================================

print("\n" + "=" * 70)
print("ARQUIVOS GERADOS")
print("=" * 70)

print(
    f"\nDataset de modelagem:"
    f"\n{caminho_saida}"
)

print(
    f"\nTabela de transformações:"
    f"\n{caminho_transformacoes}"
)

print("\n>>> TRATAMENTO DOS PI CONCLUÍDO <<<")


