import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Geração de dados simulados com NumPy
np.random.seed(42)
x = np.linspace(0, 10, 100)
y_real = 2.5 * np.sin(1.5 * x) + 5
y_ruido = y_real + np.random.normal(0, 0.5, size=x.shape)

# Organização e manipulação dos dados com Pandas
df = pd.DataFrame({
    'x': x,
    'y_observado': y_ruido
})

# Ajuste de curva (modelagem) utilizando SciPy
def func_modelo(x, a, b, c):
    return a * np.sin(b * x) + c

parametros_otimizados, _ = curve_fit(func_modelo, df['x'], df['y_observado'])
df['y_ajustado'] = func_modelo(df['x'], *parametros_otimizados)

# Visualização gráfica com Matplotlib
plt.figure(figsize=(9, 5))
plt.scatter(df['x'], df['y_observado'], color='gray', alpha=0.7, label='Dados Observados')
plt.plot(df['x'], df['y_ajustado'], color='red', linewidth=2, label=f'Ajuste SciPy: {parametros_otimizados[0]:.2f}sin({parametros_otimizados[1]:.2f}x) + {parametros_otimizados[2]:.2f}')
plt.plot(df['x'], y_real, color='blue', linestyle='--', alpha=0.5, label='Função Real Ideal')

plt.title('Ajuste de Curva e Análise de Dados com Stack Científica')
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()