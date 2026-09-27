# -*- coding: utf-8 -*-

"""
============================================================
GERADOR BUCKINGHAM-PI
============================================================

"""

# ============================================================
# IMPORTAÇÕES
# ============================================================

import os
from fractions import Fraction

import numpy as np
import pandas as pd
from scipy.linalg import null_space


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

arq_data = "Data_set.CSV"
arq_param = "Data_set_param.CSV"

caminho_data = os.path.join(
    diretorio,
    arq_data
)

caminho_param = os.path.join(
    diretorio,
    arq_param
)


# ============================================================
# 2. LEITURA DOS ARQUIVOS
# ============================================================

print("=" * 70)
print("LEITURA")
print("=" * 70)

data = pd.read_csv(
    caminho_data,
    sep=";",
    decimal=",",
    encoding="latin-1"
)

param = pd.read_csv(
    caminho_param,
    sep=";",
    decimal=",",
    encoding="latin-1"
)

# Limpar espaços acidentais nos nomes das colunas
data.columns = (
    data.columns
    .astype(str)
    .str.strip()
)

param.columns = (
    param.columns
    .astype(str)
    .str.strip()
)

print(
    f"Dataset: {data.shape[0]} linhas x "
    f"{data.shape[1]} colunas"
)

print(
    f"Parâmetros: {param.shape[0]} linhas x "
    f"{param.shape[1]} colunas"
)


# ============================================================
# 3. VALIDAR TABELA DE PARÂMETROS
# ============================================================

print("\n" + "=" * 70)
print("VALIDAÇÃO DOS PARÂMETROS")
print("=" * 70)

colunas_obrigatorias = [
    "Alias",
    "Colunas - Parametros",
    "tipo",
    "LM",
    "LD",
    "LT",
    "LK"
]

faltantes_param = [
    coluna
    for coluna in colunas_obrigatorias
    if coluna not in param.columns
]

if faltantes_param:

    print("\nERRO: colunas ausentes na tabela de parâmetros:")

    for coluna in faltantes_param:
        print(f"  - {coluna}")

    raise KeyError(
        "Tabela de parâmetros incompleta."
    )


# ============================================================
# 4. VARIÁVEIS
# ============================================================

variables = (
    param["Alias"]
    .astype(str)
    .str.strip()
    .tolist()
)

parameter_columns = (
    param["Colunas - Parametros"]
    .astype(str)
    .str.strip()
    .tolist()
)

types = (
    param["tipo"]
    .astype(str)
    .str.strip()
    .tolist()
)


print("\n" + "=" * 70)
print("VARIÁVEIS")
print("=" * 70)

for i, (alias, tipo) in enumerate(
    zip(variables, types)
):

    print(
        f"{i:2d} -> "
        f"{alias:10s} [{tipo}]"
    )


# ============================================================
# 5. MATRIZ DIMENSIONAL A
# ============================================================

A = np.array([
    pd.to_numeric(param["LM"]),
    pd.to_numeric(param["LD"]),
    pd.to_numeric(param["LT"]),
    pd.to_numeric(param["LK"])
], dtype=float)


print("\n" + "=" * 70)
print("MATRIZ DIMENSIONAL A")
print("=" * 70)

print(A)


# ============================================================
# 6. RANK
# ============================================================

rank = np.linalg.matrix_rank(A)

numero_variaveis = A.shape[1]

expected_pi = numero_variaveis - rank

print("\n" + "=" * 70)
print("DIMENSÃO")
print("=" * 70)

print(f"Variáveis = {numero_variaveis}")
print(f"Rank(A)   = {rank}")
print(f"Pi esperados = {expected_pi}")


# ============================================================
# 7. NULL SPACE
# ============================================================

N_scipy = null_space(A)

nullity = N_scipy.shape[1]

print("\n" + "=" * 70)
print("NULL SPACE - SCIPY")
print("=" * 70)

print(f"Dimensão do Null Space = {nullity}")
print(f"Esperado               = {expected_pi}")


# ============================================================
# 8. VERIFICAR BASE DO SCIPY
# ============================================================

print("\nVerificando A @ X:")

for j in range(nullity):

    X = N_scipy[:, j]

    residual = A @ X

    erro = np.max(
        np.abs(residual)
    )

    print(
        f"X{j + 1:02d} -> "
        f"erro máximo = {erro:.3e}"
    )


# ============================================================
# 9. ENCONTRAR COLUNAS INDEPENDENTES
# ============================================================

def independent_columns(
    A,
    tolerance=1e-12
):

    selected = []
    current_rank = 0

    rank_total = np.linalg.matrix_rank(
        A,
        tol=tolerance
    )

    for j in range(A.shape[1]):

        candidate = A[
            :,
            selected + [j]
        ]

        candidate_rank = np.linalg.matrix_rank(
            candidate,
            tol=tolerance
        )

        if candidate_rank > current_rank:

            selected.append(j)
            current_rank = candidate_rank

        if current_rank == rank_total:
            break

    return selected


pivot_cols = independent_columns(A)

free_cols = [
    j
    for j in range(A.shape[1])
    if j not in pivot_cols
]


print("\n" + "=" * 70)
print("ESTRUTURA DA BASE")
print("=" * 70)

print("\nColunas independentes:")

for j in pivot_cols:
    print(
        f"{j:2d} -> {variables[j]}"
    )

print("\nVariáveis livres:")

for j in free_cols:
    print(
        f"{j:2d} -> {variables[j]}"
    )


# ============================================================
# 10. CONSTRUIR N LEGÍVEL
# ============================================================

def build_nullspace_basis(
    A,
    pivot_cols,
    free_cols
):

    Ap = A[:, pivot_cols]

    basis = []

    for free_col in free_cols:

        X = np.zeros(
            A.shape[1],
            dtype=float
        )

        # variável livre = 1
        X[free_col] = 1.0

        Af = A[:, free_col]

        # Ap * Xp = -Af
        Xp = np.linalg.solve(
            Ap,
            -Af
        )

        X[pivot_cols] = Xp

        # Limpar ruído numérico
        X[
            np.abs(X) < 1e-12
        ] = 0.0

        basis.append(X)

    return np.column_stack(basis)


N = build_nullspace_basis(
    A,
    pivot_cols,
    free_cols
)


# ============================================================
# 11. NOMES DOS PI
# ============================================================

Pi_names = [
    f"Pi_{j + 1}"
    for j in range(N.shape[1])
]


# ============================================================
# 12. MATRIZ N COMO DATAFRAME
# ============================================================

Pi_matrix = pd.DataFrame(
    N,
    index=variables,
    columns=Pi_names
)

Pi_matrix[
    np.abs(Pi_matrix) < 1e-12
] = 0.0


print("\n" + "=" * 70)
print("MATRIZ N")
print("=" * 70)

print(Pi_matrix)


# ============================================================
# 13. RELAÇÕES DOS PI
# ============================================================

Pi_relations = {}

for j, pi_name in enumerate(Pi_names):

    X = N[:, j]

    relations = {}

    for variable, exponent in zip(
        variables,
        X
    ):

        if abs(exponent) > 1e-12:

            frac = Fraction(
                float(exponent)
            ).limit_denominator(24)

            relations[variable] = frac

    Pi_relations[pi_name] = relations


# ============================================================
# 14. EXPRESSÕES DOS PI
# ============================================================

def build_pi_expression(X):

    numerator = []
    denominator = []

    for variable, exponent in zip(
        variables,
        X
    ):

        if abs(exponent) < 1e-12:
            continue

        frac = Fraction(
            float(exponent)
        ).limit_denominator(24)

        # ------------------------------
        # Expoente positivo
        # ------------------------------

        if frac > 0:

            if frac == 1:

                numerator.append(variable)

            else:

                numerator.append(
                    f"{variable}^{frac}"
                )

        # ------------------------------
        # Expoente negativo
        # ------------------------------

        else:

            frac = abs(frac)

            if frac == 1:

                denominator.append(variable)

            else:

                denominator.append(
                    f"{variable}^{frac}"
                )

    if not numerator:

        numerator = ["1"]

    expression = " * ".join(
        numerator
    )

    if denominator:

        expression += (
            " / ("
            + " * ".join(denominator)
            + ")"
        )

    return expression


Pi_expressions = {}

for j, pi_name in enumerate(Pi_names):

    Pi_expressions[pi_name] = (
        build_pi_expression(N[:, j])
    )


# ============================================================
# 15. MOSTRAR OS PI
# ============================================================

print("\n" + "=" * 70)
print("GRUPOS PI")
print("=" * 70)

for pi_name in Pi_names:

    print(
        f"{pi_name} = "
        f"{Pi_expressions[pi_name]}"
    )


# ============================================================
# 16. MAPA ALIAS -> COLUNA DO DATASET
# ============================================================

experimentos = [
    "D1T1A",
    "D1T1B",
    "D1T2",
    "D2T1"
]

data_map = {}

for experimento in experimentos:

    data_map[experimento] = {}

    for alias, parameter_column in zip(
        variables,
        parameter_columns
    ):

        coluna_real = (
            f"{experimento} - "
            f"{parameter_column}"
        )

        if coluna_real in data.columns:

            data_map[
                experimento
            ][alias] = coluna_real

        else:

            data_map[
                experimento
            ][alias] = None


# ============================================================
# 17. VALIDAR OS MAPAS
# ============================================================

print("\n" + "=" * 70)
print("VALIDAÇÃO DOS ENSAIOS")
print("=" * 70)

for experimento in experimentos:

    faltantes = [
        alias
        for alias in variables
        if data_map[
            experimento
        ][alias] is None
    ]

    if faltantes:

        print(
            f"\n{experimento}: ERRO"
        )

        for alias in faltantes:

            print(
                f"  Ausente -> {alias}"
            )

    else:

        print(
            f"{experimento}: "
            f"{len(variables)} variáveis encontradas"
        )


# ============================================================
# 18. PREPARAR UM ENSAIO
# ============================================================

def prepare_experiment(
    data,
    data_map,
    experiment,
    variables
):

    """
    Cria um DataFrame usando os Aliases.

    IMPORTANTE:
    Remove somente as linhas onde TODAS as
    variáveis estão NaN.

    Portanto:

    padding -> removido

    NaN real de uma variável -> preservado
    """

    trial = pd.DataFrame(
        index=data.index
    )

    for alias in variables:

        coluna_real = (
            data_map[
                experiment
            ][alias]
        )

        if coluna_real is None:

            raise KeyError(
                f"{experiment}: "
                f"coluna não encontrada "
                f"para {alias}"
            )

        trial[alias] = pd.to_numeric(
            data[coluna_real],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Remover apenas padding
    # --------------------------------------------------------

    trial = trial.dropna(
        how="all"
    )

    # --------------------------------------------------------
    # Novo índice interno do ensaio
    # --------------------------------------------------------

    trial = trial.reset_index(
        drop=True
    )

    return trial


# ============================================================
# 19. PREPARAR D1T1A
# ============================================================

experimento_teste = "D1T1A"

print("\n" + "=" * 70)
print(
    f"PREPARANDO ENSAIO -> "
    f"{experimento_teste}"
)
print("=" * 70)

D1T1A_data = prepare_experiment(
    data,
    data_map,
    experimento_teste,
    variables
)

print(
    f"Número de amostras reais = "
    f"{len(D1T1A_data)}"
)


# ============================================================
# 20. DIAGNÓSTICO DOS DADOS
# ============================================================

print("\n" + "=" * 70)
print("DIAGNÓSTICO DOS DADOS - D1T1A")
print("=" * 70)

for variable in variables:

    values = D1T1A_data[
        variable
    ].to_numpy(
        dtype=float
    )

    n_nan = np.sum(
        np.isnan(values)
    )

    n_inf = np.sum(
        np.isinf(values)
    )

    n_zero = np.sum(
        values == 0
    )

    print(
        f"{variable:10s} | "
        f"NaN = {n_nan:4d} | "
        f"Inf = {n_inf:4d} | "
        f"Zero = {n_zero:4d}"
    )


# ============================================================
# 21. CALCULAR OS PI
# ============================================================

def calculate_pis(
    trial_data,
    variables,
    N
):

    """
    Calcula os grupos Pi.

    Regras:

    valor NaN:
        -> Pi = NaN

    valor Inf:
        -> Pi = NaN

    zero com expoente positivo:
        -> permite zero

    zero com expoente negativo:
        -> Pi = NaN
    """

    Q = trial_data[
        variables
    ].to_numpy(
        dtype=float
    )

    n_samples = Q.shape[0]
    n_pis = N.shape[1]

    Pi_values = np.full(
        (n_samples, n_pis),
        np.nan,
        dtype=float
    )

    # ========================================================
    # Pi por Pi
    # ========================================================

    for j in range(n_pis):

        X = N[:, j]

        pi = np.ones(
            n_samples,
            dtype=float
        )

        valid = np.ones(
            n_samples,
            dtype=bool
        )

        # ----------------------------------------------------
        # Aplicar cada variável participante
        # ----------------------------------------------------

        for i, exponent in enumerate(X):

            if abs(exponent) < 1e-12:
                continue

            values = Q[:, i]

            # -----------------------------------------------
            # NaN ou Inf
            # -----------------------------------------------

            invalid = (
                np.isnan(values)
                | np.isinf(values)
            )

            valid &= ~invalid

            # -----------------------------------------------
            # Zero com expoente negativo
            # -----------------------------------------------

            if exponent < 0:

                valid &= (
                    values != 0
                )

            # -----------------------------------------------
            # Calcular somente onde ainda é válido
            # -----------------------------------------------

            with np.errstate(
                divide="ignore",
                invalid="ignore",
                over="ignore"
            ):

                pi[valid] *= (
                    values[valid]
                    ** exponent
                )

        # ----------------------------------------------------
        # Onde não foi possível calcular -> NaN
        # ----------------------------------------------------

        pi[~valid] = np.nan

        Pi_values[:, j] = pi

    # ========================================================
    # DataFrame
    # ========================================================

    return pd.DataFrame(
        Pi_values,
        index=trial_data.index,
        columns=[
            f"Pi_{j + 1}"
            for j in range(n_pis)
        ]
    )


# ============================================================
# 22. CALCULAR OS PI E GERAR DIAGNÓSTICO
# ============================================================

def calculate_pis_with_diagnostic(
    trial_data,
    variables,
    N,
    Pi_expressions,
    experiment
):
    """
    Calcula todos os grupos Pi de um ensaio.

    Retorna:
        Pi_data
            DataFrame com os valores dos Pi.

        diagnostic_data
            DataFrame contendo apenas os casos em que
            um Pi não pôde ser calculado.
    """

    Q = trial_data[
        variables
    ].to_numpy(dtype=float)

    n_samples = Q.shape[0]
    n_pis = N.shape[1]

    Pi_values = np.full(
        (n_samples, n_pis),
        np.nan,
        dtype=float
    )

    diagnostic_rows = []

    # ========================================================
    # Pi por Pi
    # ========================================================

    for j in range(n_pis):

        pi_name = f"Pi_{j + 1}"

        X = N[:, j]

        pi = np.ones(
            n_samples,
            dtype=float
        )

        valid = np.ones(
            n_samples,
            dtype=bool
        )

        # Uma lista de motivos para cada amostra
        reasons = [
            []
            for _ in range(n_samples)
        ]

        # ----------------------------------------------------
        # Analisar cada variável participante
        # ----------------------------------------------------

        for i, exponent in enumerate(X):

            # Variável não participa do Pi
            if abs(exponent) < 1e-12:
                continue

            variable = variables[i]

            values = Q[:, i]

            # -----------------------------------------------
            # NaN
            # -----------------------------------------------

            mask_nan = np.isnan(values)

            valid &= ~mask_nan

            for idx in np.where(mask_nan)[0]:

                reasons[idx].append(
                    f"{variable}=NaN"
                )

            # -----------------------------------------------
            # Inf
            # -----------------------------------------------

            mask_inf = np.isinf(values)

            valid &= ~mask_inf

            for idx in np.where(mask_inf)[0]:

                reasons[idx].append(
                    f"{variable}=Inf"
                )

            # -----------------------------------------------
            # Zero no denominador
            # -----------------------------------------------

            if exponent < 0:

                mask_zero_den = (
                    values == 0
                )

                valid &= ~mask_zero_den

                for idx in np.where(
                    mask_zero_den
                )[0]:

                    reasons[idx].append(
                        f"{variable}=0 "
                        f"(denominador)"
                    )

            # -----------------------------------------------
            # Calcular somente valores válidos
            # -----------------------------------------------

            with np.errstate(
                divide="ignore",
                invalid="ignore",
                over="ignore"
            ):

                pi[valid] *= (
                    values[valid]
                    ** exponent
                )

        # ----------------------------------------------------
        # Verificar resultado final
        # ----------------------------------------------------

        result_invalid = (
            valid
            &
            ~np.isfinite(pi)
        )

        valid &= ~result_invalid

        for idx in np.where(
            result_invalid
        )[0]:

            reasons[idx].append(
                "resultado não finito"
            )

        # ----------------------------------------------------
        # Amostras inválidas -> NaN
        # ----------------------------------------------------

        pi[~valid] = np.nan

        Pi_values[:, j] = pi

        # ----------------------------------------------------
        # Registrar diagnóstico
        # ----------------------------------------------------

        for idx in np.where(
            ~valid
        )[0]:

            diagnostic_rows.append({

                "Experimento":
                    experiment,

                "Amostra":
                    idx,

                "Temp":
                    trial_data.loc[
                        idx,
                        "Temp"
                    ],

                "Pi":
                    pi_name,

                "Expressao":
                    Pi_expressions[
                        pi_name
                    ],

                "Motivo":
                    "; ".join(
                        reasons[idx]
                    )
            })

    # ========================================================
    # DataFrame dos Pi
    # ========================================================

    Pi_data = pd.DataFrame(
        Pi_values,
        index=trial_data.index,
        columns=Pi_names
    )

    # ========================================================
    # DataFrame do diagnóstico
    # ========================================================

    diagnostic_data = pd.DataFrame(
        diagnostic_rows
    )

    return (
        Pi_data,
        diagnostic_data
    )


# ============================================================
# 23. PROCESSAR OS QUATRO ENSAIOS
# ============================================================

Pi_data = {}

Experiment_data = {}

Diagnostics = []

for experimento in experimentos:

    print("\n" + "=" * 70)
    print(
        f"PROCESSANDO {experimento}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Preparar ensaio
    # --------------------------------------------------------

    trial = prepare_experiment(
        data,
        data_map,
        experimento,
        variables
    )

    # --------------------------------------------------------
    # Calcular Pi + diagnóstico
    # --------------------------------------------------------

    Pi, diagnostic = (
        calculate_pis_with_diagnostic(
            trial,
            variables,
            N,
            Pi_expressions,
            experimento
        )
    )

    # --------------------------------------------------------
    # Guardar
    # --------------------------------------------------------

    Experiment_data[
        experimento
    ] = trial

    Pi_data[
        experimento
    ] = Pi

    Diagnostics.append(
        diagnostic
    )

    print(
        f"Amostras = {len(trial)}"
    )

    print(
        f"Pi = {Pi.shape[1]}"
    )

    print(
        f"Diagnósticos = "
        f"{len(diagnostic)}"
    )


# ============================================================
# 24. MONTAR DATASET DIMENSIONAL FINAL
# ============================================================

dataset_dimensional = []

for experimento in experimentos:

    trial = Experiment_data[
        experimento
    ]

    Pi = Pi_data[
        experimento
    ].copy()

    # --------------------------------------------------------
    # Metadados
    # --------------------------------------------------------

    Pi.insert(
        0,
        "Temp",
        trial["Temp"].to_numpy()
    )

    Pi.insert(
        0,
        "Amostra",
        np.arange(len(trial))
    )

    Pi.insert(
        0,
        "Experimento",
        experimento
    )

    dataset_dimensional.append(
        Pi
    )


dataset_dimensional = pd.concat(
    dataset_dimensional,
    ignore_index=True
)


# ============================================================
# 25. NÚMERO DE PI VÁLIDOS POR AMOSTRA
# ============================================================

dataset_dimensional[
    "N_Pi_validos"
] = (
    dataset_dimensional[
        Pi_names
    ]
    .notna()
    .sum(axis=1)
)


# ============================================================
# 26. DATASET DE DIAGNÓSTICO
# ============================================================

if Diagnostics:

    dataset_diagnostico = pd.concat(
        Diagnostics,
        ignore_index=True
    )

else:

    dataset_diagnostico = pd.DataFrame(
        columns=[
            "Experimento",
            "Amostra",
            "Temp",
            "Pi",
            "Expressao",
            "Motivo"
        ]
    )


# ============================================================
# 27. RESUMO DO DATASET DIMENSIONAL
# ============================================================

print("\n" + "=" * 70)
print("DATASET DIMENSIONAL FINAL")
print("=" * 70)

print(
    f"Linhas  = "
    f"{dataset_dimensional.shape[0]}"
)

print(
    f"Colunas = "
    f"{dataset_dimensional.shape[1]}"
)

print("\nAmostras por ensaio:")

print(
    dataset_dimensional[
        "Experimento"
    ]
    .value_counts()
    .sort_index()
)


print("\nPrimeiras linhas:")

print(
    dataset_dimensional.head(
        10
    ).to_string()
)


# ============================================================
# 28. RESUMO DO DIAGNÓSTICO
# ============================================================

print("\n" + "=" * 70)
print("DATASET DE DIAGNÓSTICO")
print("=" * 70)

print(
    f"Registros de diagnóstico = "
    f"{len(dataset_diagnostico)}"
)

if len(dataset_diagnostico) > 0:

    print("\nPrimeiros registros:")

    print(
        dataset_diagnostico.head(
            20
        ).to_string(
            index=False
        )
    )


# ============================================================
# 29. RESUMO POR ENSAIO E PI
# ============================================================

diagnostic_summary = (
    dataset_diagnostico
    .groupby(
        [
            "Experimento",
            "Pi"
        ],
        dropna=False
    )
    .size()
    .reset_index(
        name="N_invalidos"
    )
)


print("\n" + "=" * 70)
print("RESUMO DOS DIAGNÓSTICOS")
print("=" * 70)

print(
    diagnostic_summary.to_string(
        index=False
    )
)


# ============================================================
# 30. RESUMO DE COBERTURA DO DATASET FINAL
# ============================================================

coverage_final = []

for experimento in experimentos:

    subset = dataset_dimensional[
        dataset_dimensional[
            "Experimento"
        ] == experimento
    ]

    for pi_name in Pi_names:

        n_total = len(subset)

        n_valid = (
            subset[pi_name]
            .notna()
            .sum()
        )

        pct_valid = (
            100
            * n_valid
            / n_total
        )

        coverage_final.append({

            "Experimento":
                experimento,

            "Pi":
                pi_name,

            "N_total":
                n_total,

            "N_valid":
                n_valid,

            "N_NaN":
                n_total - n_valid,

            "Pct_valid":
                pct_valid
        })


coverage_final = pd.DataFrame(
    coverage_final
)


print("\n" + "=" * 70)
print("COBERTURA FINAL")
print("=" * 70)

print(
    coverage_final.to_string(
        index=False
    )
)


# ============================================================
# 31. SALVAR OS DATASETS
# ============================================================

arquivo_dimensional = os.path.join(
    diretorio,
    "Dataset_Dimensional_Final.csv"
)

arquivo_diagnostico = os.path.join(
    diretorio,
    "Dataset_Diagnostico_Pi.csv"
)

arquivo_cobertura = os.path.join(
    diretorio,
    "Dataset_Cobertura_Pi.csv"
)

arquivo_Expressoes = os.path.join(
    diretorio,
    "Dataset_Expressoes_Pi.csv"
)

dataset_dimensional.to_csv(
    arquivo_dimensional,
    sep=";",
    decimal=",",
    encoding="latin-1",
    index=False
)

dataset_diagnostico.to_csv(
    arquivo_diagnostico,
    sep=";",
    decimal=",",
    encoding="latin-1",
    index=False
)

coverage_final.to_csv(
    arquivo_cobertura,
    sep=";",
    decimal=",",
    encoding="latin-1",
    index=False
)

(pd.DataFrame(list(Pi_expressions.items()), columns=['PI', 'Expressão'])).to_csv(
    arquivo_Expressoes,
    sep=";",
    decimal=",",
    encoding="latin-1",
    index=False
)


# ============================================================
# 32. FINAL
# ============================================================

print("\n" + "=" * 70)
print("ARQUIVOS GERADOS")
print("=" * 70)

print(
    f"\nDataset dimensional:"
    f"\n{arquivo_dimensional}"
)

print(
    f"\nDiagnóstico:"
    f"\n{arquivo_diagnostico}"
)

print(
    f"\nCobertura:"
    f"\n{arquivo_cobertura}"
)

print("\n>>> ETAPA DIMENSIONAL CONCLUÍDA <<<")