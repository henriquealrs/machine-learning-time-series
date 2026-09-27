
import os

# ============================================================
# 23. ABRIR O PROGRAMA DE PLOTS
# ============================================================
pasta_atual = os.getcwd()
caminho_alvo = (os.path.join(pasta_atual, "Auxi" )+"\\" ).replace("\\", "/")

diretorio_main = ( caminho_alvo
    
)

# -------------------------------

arquivo_leitura = os.path.join(
    diretorio_main,
    "New_motor_in_car2.py"
).replace("\\", "/")

print("\n" + "=" * 70)
print("1 ------- EXECUTANDO PROGRAMA LEITURA")
print("=" * 70)
print("\n")


exec(
    open(
        arquivo_leitura,
        encoding="utf-8"
    ).read()
)

# -------------------------------

arquivo_plot1 = os.path.join(
    diretorio_main,
    "Plot_Pis.py"
).replace("\\", "/")

print("\n" + "=" * 70)
print("1 ------- EXECUTANDO PROGRAMA PLOT 1")
print("=" * 70)
print("\n")


exec(
    open(
        arquivo_plot1,
        encoding="utf-8"
    ).read()
)

# -------------------------------

arquivo_tratamento = os.path.join(
    diretorio_main,
    "Tratamento.py"
).replace("\\", "/")

print("\n" + "=" * 70)
print("1 ------- EXECUTANDO PROGRAMA TRATAMENTO DOS DADOS")
print("=" * 70)
print("\n")


exec(
    open(
        arquivo_tratamento,
        encoding="utf-8"
    ).read()
)

# -------------------------------

arquivo_plot2 = os.path.join(
    diretorio_main,
    "Plot_Pis2.py"
).replace("\\", "/")

print("\n" + "=" * 70)
print("1 ------- EXECUTANDO PROGRAMA PLOT 2")
print("=" * 70)
print("\n")


exec(
    open(
        arquivo_plot2,
        encoding="utf-8"
    ).read()
)
