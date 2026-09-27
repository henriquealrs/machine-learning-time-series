# Sugestões de machine learning para os dados do motor

## Leitura dos dados e pergunta central

Os dados em `Dados/Data_set.CSV` contêm quatro ensaios independentes (`D1T1A`, `D1T1B`, `D1T2` e `D2T1`) organizados lado a lado. Há 3.876 linhas no arquivo largo, mas cada ensaio tem, respectivamente, 1.984, 1.776, 3.876 e 1.703 amostras. A linha 100 de um ensaio não deve ser interpretada como simultânea à linha 100 de outro. Para modelar, empilhem os ensaios e preservem `Experimento` e `Temp`.

A tabela `Dados/Data_set_param.CSV` classifica `Acc`, `Brake` e `Gear` como variáveis de **controle**. Classifica `Tau` (torque real do motor, N·m), `n` (rotação, 1/s), `Velo` (m/s), `Taccm` (taxa de consumo), `TeLam`, `Teom`, `Team`, `Diem`, `x`, `zAl`, `yLa`, `xLo` e `mass` como **físicas**. `Temp` é a coordenada temporal. Essa separação sugere uma pergunta útil: **em quais condições de comando e operação o torque `Tau` e o consumo se elevam?**

Não existe coluna de severidade ou de falha confirmada. Por isso, um grupo ou anomalia só pode ser chamado de padrão operacional até que exista uma definição de severidade validada tecnicamente.

### Esclarecimento sobre `Diem` — 27/09/2026

**`Diem` representa tempo de abertura da injeção + pressão de injeção**. O “+” indica a participação das duas grandezas na descrição da injeção; ele não define uma soma numérica entre tempo e pressão, que têm dimensões diferentes.

Nos arquivos disponíveis há uma única coluna `Diem` por ensaio, identificada como m³/s. A planilha também menciona uma grandeza de origem em mm³/ciclo. Não há colunas separadas de duração de abertura e pressão nos cabeçalhos dos dados. Portanto, a definição conceitual está registrada, mas ainda falta confirmar **como duração e pressão foram transformadas no valor escalar de `Diem`**, suas unidades e se o registro é medição, comando ou valor calculado.

- **Previsão supervisionada:** uma próxima comparação pode acrescentar `Diem(t)` e seu histórico para prever consumo futuro, usando somente D1T2 e D2T1, onde a variável existe. Comparar modelos com e sem `Diem` nas mesmas janelas, treinando em um ensaio e testando no outro, sem seleção de parâmetros no teste. Não usar `Diem` futuro nem preencher ensaios inteiros ausentes com zero. Com apenas dois ensaios, a conclusão será exploratória. Essa comparação ainda não foi executada.
- **Análise não supervisionada:** nesses mesmos dois ensaios, explorar agrupamentos de `Diem`, `Tau`, `n` e consumo, com padronização e comparação da composição por ensaio. Os grupos indicariam regimes observados de injeção/operação; não identificariam separadamente o efeito da pressão e da abertura.
- **Controle:** para testar duração de abertura e pressão como duas ações, precisamos das duas séries separadas, unidades, limites, sincronização e identificação de comandos versus medições. Uma única coluna composta não permite recuperar de forma única esses dois valores. A simulação executada continua atuando sobre `Acc`; a definição recebida não permite transformá-la em controle de injeção.
- **`Pi_5`:** a expressão `Diem / (Temp² × Velo³)` só é adimensional sob a dimensão de vazão (L³/T) declarada na tabela atual. A nova descrição conceitual não confirma essa dimensão. Se `Diem` tiver outra dimensão ou for desdobrada em duração e pressão, será necessário corrigir os metadados e recalcular os grupos Pi. Os Pi existentes mantêm a hipótese documental original.

A unidade/calibração de `Taccm` continua pendente. Os resultados já executados não usam `Diem` nem `Pi_5` como entrada e não mudam com esse esclarecimento.

### Cuidados identificados

- `Diem` está inteiramente ausente em `D1T1A` e `D1T1B`. Em `D2T1`, `Teom`, `yLa`, `xLo` e `mass` estão inteiramente ausentes.
- O freio chega a 102%; a altitude em `D2T1` chega a 8.291 m. Verifiquem unidades, saturação e erros de sensor antes de interpretar esses pontos.
- `Dados/Data_set.CSV` e `Data_set limpo/Data_set.CSV` são idênticos byte a byte.
- A execução de `Main_Motor.py` gerou os grupos Buckingham Pi, as bases derivadas, diagnósticos de cobertura e gráficos em `Dados/`.

## Preparação reutilizável

O código a seguir pode abrir um notebook ou script executado da raiz do repositório. O CSV usa `;`, vírgula decimal e `latin-1`. Os aliases são lidos da tabela de parâmetros na mesma ordem das 17 colunas de cada ensaio.

```python
from pathlib import Path
import pandas as pd

pasta = Path("Motor/Dados")
bruto = pd.read_csv(pasta / "Data_set.CSV", sep=";", decimal=",", encoding="latin-1")
param = pd.read_csv(pasta / "Data_set_param.CSV", sep=";", decimal=",", encoding="latin-1")
aliases = param["Alias"].str.strip().tolist()
ensaios = ["D1T1A", "D1T1B", "D1T2", "D2T1"]

partes = []
for ensaio in ensaios:
    colunas = [c for c in bruto.columns if c.startswith(f"{ensaio} - ")]
    assert len(colunas) == len(aliases), ensaio
    parte = bruto[colunas].copy()
    parte.columns = aliases
    parte = parte.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    parte["Experimento"] = ensaio
    partes.append(parte)

dados = pd.concat(partes, ignore_index=True)
dados = dados.sort_values(["Experimento", "Temp"]).reset_index(drop=True)
assert not dados.duplicated(["Experimento", "Temp"]).any()

# Presentes nos quatro ensaios; Tau entra explicitamente.
fisicas = ["Velo", "n", "Tau", "TeLam", "Team"]
controles = ["Acc", "Brake", "Gear"]
atributos_regime = controles + fisicas
print(dados.groupby("Experimento").size())
print(dados[atributos_regime + ["Taccm"]].isna().mean())
```

Antes de treinar, tracem `Tau`, `n`, `Velo`, `Acc`, `Brake` e `Taccm` ao longo de `Temp` para cada ensaio. Confirmem se `Temp` equivale de fato a segundos antes de interpretar dez amostras como dez segundos.

## Resultados reais da execução de `Main_Motor.py`

O programa concluiu a geração de:

- `Dados/Dataset_Dimensional_Final.csv`: 9.339 amostras dos quatro ensaios, 13 Pi e metadados;
- `Dados/Dataset_Modelagem_Pi.csv`: as mesmas 9.339 amostras, mais 7 transformações de Pi;
- `Dados/Dataset_Expressoes_Pi.csv`, `Dataset_Cobertura_Pi.csv`, `Dataset_Diagnostico_Pi.csv` e `Dataset_Transformacoes_Pi.csv`;
- 13 gráficos em `Dados/Graficos_Pi/` e 7 em `Dados/Graficos_Modelagem_Pi/`.

As expressões abaixo foram lidas de `Dataset_Expressoes_Pi.csv`, não deduzidas a partir do nome:

| Pi | Expressão gerada | Leitura para o projeto |
| --- | --- | --- |
| `Pi_1`, `Pi_2`, `Pi_3` | `Acc`, `Brake`, `Gear` | São os próprios comandos, sem informação nova em relação aos sensores. |
| `Pi_4` | `Temp × n` | Combina tempo decorrido e rotação; pode refletir a duração do ensaio. |
| `Pi_5` | `Diem / (Temp² × Velo³)` | Injeção normalizada; adimensional apenas se `Diem` for vazão (L³/T). |
| `Pi_6` | `Taccm / (Temp² × Velo³)` | Consumo ajustado por tempo e velocidade. |
| `Pi_7`, `Pi_8` | `Teom / TeLam`, `Team / TeLam` | Razões térmicas; `Pi_7_rel = Pi_7 - 1` mede diferença relativa entre óleo e arrefecimento. |
| `Pi_9` | `x / (Temp × Velo)` | Relação entre distância e deslocamento esperado. |
| `Pi_10`, `Pi_11`, `Pi_12` | `zAl`, `yLa`, `xLo`, respectivamente, divididos por `Temp × Velo` | Relações de altitude e deslocamentos. |
| `Pi_13` | `Velo² × mass / Tau` | Relação de carga/movimento com torque; incorpora `Tau` diretamente. |

Cobertura de valores válidos no CSV dimensional, por ensaio (%):

| Pi | D1T1A | D1T1B | D1T2 | D2T1 |
| --- | ---: | ---: | ---: | ---: |
| `Pi_1` a `Pi_4`, `Pi_8` | 100 | 100 | 100 | 100 |
| `Pi_5` | 0 | 0 | 84,8 | 93,7 |
| `Pi_6`, `Pi_9`, `Pi_10` | 82,6 | 83,0 | 84,8 | 93,7 |
| `Pi_7` | 100 | 100 | 100 | 0 |
| `Pi_11`, `Pi_12` | 82,6 | 83,0 | 84,8 | 0 |
| `Pi_13` | 75,7 | 67,3 | 71,8 | 0 |

`Pi_5` perde os dois primeiros ensaios por ausência de `Diem`. `Pi_7`
e `Pi_13` perdem `D2T1` por ausência de `Teom` e `mass`. `Pi_6`,
`Pi_9` a `Pi_12` também dependem de velocidade não nula. As transformações
restringem ainda mais a cobertura: `Pi_5_log10` tem 3.141 valores, `Pi_6_log10`
tem 5.203 e `Pi_13_log10` tem 4.245. O log só é calculado para valores
estritamente positivos; valores zero não são erros de leitura por si sós.

## Sugestões não supervisionadas

| Pergunta | Método | Saída e avaliação |
| --- | --- | --- |
| Quais regimes de operação têm torque e consumo diferentes? | KMeans sobre sensores ou Pi | Perfis, gráficos ao longo do tempo e proporção por ensaio |
| Que trechos diferem do funcionamento usual? | IsolationForest | Alertas para inspeção física; não são falhas confirmadas |

### Regimes com sensores e `Tau`

A pergunta é se comandos e condições físicas formam regimes distintos, por exemplo parada, aceleração e alta carga. `Tau` participa diretamente da formação dos grupos. Uma alternativa é agrupar sem `Tau` e comparar o torque entre grupos; isso testa se os outros sensores distinguem regimes de torque.

```python
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

agrupador = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    KMeans(n_clusters=4, n_init=10, random_state=42),
)
dados["regime"] = agrupador.fit_predict(dados[atributos_regime])
print(dados.groupby("regime")[atributos_regime + ["Taccm"]].median().round(2))
print(pd.crosstab(dados["Experimento"], dados["regime"], normalize="index").round(2))
```

Compare 2 a 6 grupos e visualize `regime` ao longo do tempo. Nomeie os grupos **depois** de olhar seus perfis. Uma pontuação silhouette pode ajudar na escolha, mas não demonstra significado físico nem severidade.

### Detecção de trechos incomuns

Um detector pode destacar combinações raras de torque, rotação, velocidade e comandos. A fração `0.02` abaixo é somente um limite de alertas escolhido para triagem: não afirma que 2% das amostras são falhas.

```python
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

detector = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    IsolationForest(contamination=0.02, random_state=42),
)
dados["incomum"] = detector.fit_predict(dados[atributos_regime]) == -1
print(dados.loc[dados["incomum"],
    ["Experimento", "Temp", "Tau", "n", "Velo", "Acc", "Brake"]].head(30))
```

Plote cada trecho e cheque se o alerta representa um evento real, transiente normal ou leitura incorreta. Para comparar trechos inteiros, resuma janelas com mediana, máximo e variação de `Tau`; janelas sobrepostas devem ficar no mesmo conjunto de avaliação.

### Regimes com grupos Pi

Uma análise prática é formar grupos com os Pi disponíveis nos quatro ensaios
e **depois** verificar como `Tau` e `Taccm` variam entre esses grupos.
Como `Pi_1` a `Pi_3` são comandos originais, esse experimento pergunta se
comandos e a razão térmica `Pi_8` identificam regimes de torque. `Pi_4`
fica fora deste primeiro teste porque contém o tempo decorrido.

```python
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd

base_pi = pd.read_csv(
    "Motor/Dados/Dataset_Modelagem_Pi.csv",
    sep=";", decimal=",", encoding="latin-1",
)
express = pd.read_csv(
    "Motor/Dados/Dataset_Expressoes_Pi.csv",
    sep=";", decimal=",", encoding="latin-1",
)
print(express.to_string(index=False))

colunas_pi = ["Pi_1", "Pi_2", "Pi_3", "Pi_8"]
modelo_pi = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    KMeans(n_clusters=4, n_init=10, random_state=42),
)
base_pi["regime_pi"] = modelo_pi.fit_predict(base_pi[colunas_pi])

# Junta o resultado aos sensores brutos pelo ensaio e pelo tempo.
comparacao = base_pi.merge(
    dados[["Experimento", "Temp", "Tau", "Taccm"]],
    on=["Experimento", "Temp"], validate="one_to_one",
)
print(comparacao.groupby("regime_pi")[colunas_pi + ["Tau", "Taccm"]].median())
print(pd.crosstab(
    comparacao["Experimento"], comparacao["regime_pi"], normalize="index"
).round(2))
```

Compare esse agrupamento ao obtido com os sensores originais. Se os grupos forem dominados por
freio ou marcha, digam isso claramente; não atribuam severidade a eles.
Um segundo experimento pode incluir `Pi_13` apenas nos três ensaios que têm
`mass` e informar quantas linhas foram excluídas. Nesse caso, o torque
participa da construção dos grupos e não pode ser apresentado como uma
descoberta independente.

### Resultado preliminar dos agrupamentos

No agrupamento com `Pi_1`, `Pi_2`, `Pi_3` e `Pi_8`, um dos quatro grupos
teve mediana de acelerador 68,8%, `Tau` 784 N·m e `Taccm` 24,75 na
unidade registrada. Outro teve mediana de `Tau` e `Taccm` igual a zero.
Isso sugere regimes de operação diferentes, mas ainda requer gráficos e
verificação da qualidade dos comandos, especialmente leituras de freio
até 102%.

### Aplicação: análise da operação

Os regimes podem ajudar a identificar combinações de comandos, torque e velocidade associadas a diferentes consumos. Essa análise foi desenvolvida no [notebook de análise da operação](notebooks/03_analise_operacao.ipynb) e no [documento de aplicações](APLICACOES_PREVISAO_CONSUMO.md#analise-operacao).

O novo experimento usa 310 janelas de 30 segundos e compara agrupamentos com sensores e com Pi. Nos grupos de sensores, as médias de consumo foram 4,37 e 15,65 na unidade registrada. São associações operacionais; não demonstram desperdício nem severidade. Os resultados completos e as limitações estão no documento vinculado.

## Sugestões supervisionadas

| Pergunta | Método | Alvo e avaliação |
| --- | --- | --- |
| É possível antecipar o torque? | Persistência, DummyRegressor e HistGradientBoostingRegressor | `Tau(t+10)`; MAE por ensaio deixado de fora |
| Os Pi melhoram a previsão de consumo? | HistGradientBoostingRegressor com Pi e com sensores | `Taccm(t+10)`; MAE nas mesmas linhas de teste |
| Uma severidade validada pode ser prevista? | DummyClassifier e RandomForestClassifier, após rotulação | Classe externa; matriz de confusão, recall crítico e macro F1 |

### Previsão de torque futuro

Aqui `Tau(t+10)` é o alvo. O modelo usa controles e grandezas físicas atuais, inclusive `Tau(t)`, que é um preditor legítimo para previsão. `Temp` e `Experimento` identificam a amostra, mas não entram como atributos. Nenhuma informação futura é usada na entrada.

```python
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline

alvo = "Tau_futuro"
horizonte = 10
previsao = dados.copy()
previsao[alvo] = previsao.groupby("Experimento")["Tau"].shift(-horizonte)
previsao = previsao.dropna(subset=["Tau", alvo])

for teste in ensaios:
    treino = previsao[previsao["Experimento"] != teste]
    validacao = previsao[previsao["Experimento"] == teste]
    if validacao.empty:
        continue

    constante = DummyRegressor(strategy="median")
    constante.fit(treino[atributos_regime], treino[alvo])
    modelo = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(random_state=42),
    )
    modelo.fit(treino[atributos_regime], treino[alvo])
    real = validacao[alvo]
    print(teste, {
        "MAE_mediana": mean_absolute_error(real, constante.predict(validacao[atributos_regime])),
        "MAE_persistencia": mean_absolute_error(real, validacao["Tau"]),
        "MAE_modelo": mean_absolute_error(real, modelo.predict(validacao[atributos_regime])),
    })
```

Repita trocando o alvo por `Taccm` deslocado dez amostras e adicione `Taccm(t)` aos atributos. Reporte MAE de cada ensaio separadamente e compare com persistência. Quatro ensaios são poucas unidades independentes; uma divisão aleatória de linhas superestimaria a capacidade de generalização.

### Previsão de consumo futuro com Pi

Além da previsão pontual abaixo, desenvolvemos aplicações em [Aplicações da previsão de consumo](APLICACOES_PREVISAO_CONSUMO.md), com um notebook executado para cada subtema. Os novos experimentos preveem médias ou integrais futuras, portanto suas métricas não são diretamente comparáveis às desta previsão pontual.

`Pi_6` contém `Taccm(t)`. Ele pode ser atributo para prever
`Taccm(t+10)`, pois usa consumo **atual**; seria vazamento se o alvo fosse
`Taccm(t)`. O exemplo conserva apenas linhas em que `Pi_6_log10` existe,
de modo que a comparação com sensores originais deve usar **exatamente as
mesmas linhas** e os mesmos ensaios de teste.

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

base_pi = pd.read_csv(
    "Motor/Dados/Dataset_Modelagem_Pi.csv",
    sep=";", decimal=",", encoding="latin-1",
)
futuro = dados[["Experimento", "Temp"] + atributos_regime + ["Taccm"]].copy()
futuro["consumo_futuro"] = (
    futuro.groupby("Experimento")["Taccm"].shift(-10)
)
avaliacao = base_pi.merge(
    futuro, on=["Experimento", "Temp"], validate="one_to_one"
).dropna(subset=["Pi_6_log10", "consumo_futuro"])

atributos_pi = ["Pi_1", "Pi_2", "Pi_3", "Pi_4", "Pi_6_log10", "Pi_8"]
atributos_brutos = atributos_regime + ["Taccm"]
for teste in ensaios:
    treino = avaliacao[avaliacao["Experimento"] != teste]
    validacao = avaliacao[avaliacao["Experimento"] == teste]
    if validacao.empty:
        continue
    modelo_pi = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(random_state=42),
    )
    modelo_bruto = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(random_state=42),
    )
    modelo_pi.fit(treino[atributos_pi], treino["consumo_futuro"])
    modelo_bruto.fit(treino[atributos_brutos], treino["consumo_futuro"])
    mae_pi = mean_absolute_error(
        validacao["consumo_futuro"],
        modelo_pi.predict(validacao[atributos_pi]),
    )
    mae_bruto = mean_absolute_error(
        validacao["consumo_futuro"],
        modelo_bruto.predict(validacao[atributos_brutos]),
    )
    mae_persistencia = mean_absolute_error(
        validacao["consumo_futuro"], validacao["Taccm"],
    )
    print(teste, "linhas:", len(validacao), {
        "MAE_persistencia": round(mae_persistencia, 3),
        "MAE_Pi": round(mae_pi, 3),
        "MAE_bruto": round(mae_bruto, 3),
    })
```

Para **estimar `Tau` no mesmo instante**, `Pi_13` vazaria a resposta,
porque a fórmula contém `Tau`. Para prever `Tau(t+10)`, o `Pi_13(t)`
seria temporalmente admissível, mas falta em todo `D2T1`; compare somente
ensaios e linhas com cobertura adequada. `Pi_4` contém `Temp`, então
avalie se sua inclusão produz apenas um atalho ligado à duração do ensaio.

#### Aplicação: orientação ao motorista

Antecipar consumo elevado pode apoiar a avaliação da condução. O [notebook de orientação ao motorista](notebooks/01_orientacao_motorista.ipynb) testa alertas para média elevada nos próximos 10 segundos. O modelo melhorou o F1 em três dos quatro ensaios, mas perdeu precisão e gerou mais falsos alertas. Isso não demonstra que orientar o motorista já produza economia. Veja [método, resultados e limites](APLICACOES_PREVISAO_CONSUMO.md#orientacao-motorista).

A extensão já executada escolhe o limiar por validação interna e agrupa avisos consecutivos. As janelas falsas positivas caíram de **113 para 79**, com aumento das falsas negativas de **35 para 50**; os avisos falsos agrupados caíram de **47 para 35**. Também foram medidos erros por contexto e tempo até a confirmação das janelas. A redução de alertas não é gratuita: veja o compromisso entre precisão e recall no documento vinculado.

#### Aplicação: planejamento energético

Prever consumo acumulado pode apoiar um orçamento de combustível por trecho. O [notebook de planejamento energético](notebooks/02_planejamento_energetico.ipynb) estima a integral dos próximos 30 segundos. O MAE por janela melhorou em três ensaios, porém o erro do total chegou a aproximadamente +27,2% em D1T2 e −30,1% em D2T1. A unidade de `Taccm` precisa ser confirmada antes de converter a integral para litros ou estimar autonomia. Veja [método, resultados e limites](APLICACOES_PREVISAO_CONSUMO.md#planejamento-energetico).

A extensão testou históricos de 10/30 s, inclusão/retirada de Pi_6 e uma correção aditiva aprendida em previsões fora da amostra do treino. **A calibração piorou o erro acumulado nos quatro ensaios.** Os dois MDs registram esse resultado; não apresentamos a correção como solução do viés. As alternativas foram comparadas nos mesmos instantes, sem usar o ensaio externo para escolhê-las.

#### Aplicação: controle automático

Um controlador futuro poderia usar previsões ao comparar comandos que atendam à demanda do veículo. Na primeira rodada, o [notebook de controle automático](notebooks/04_controle_automatico.ipynb) avaliou o preditor e sua sensibilidade a cenários de acelerador ±10 pontos percentuais, com atributos derivados consistentes. Dos 782 cenários, 567 passaram por um filtro de proximidade ao treino; nenhum de D2T1 passou. Mudança na previsão não é economia comprovada, e o notebook não regula a injeção. Veja [método, resultados e requisitos adicionais](APLICACOES_PREVISAO_CONSUMO.md#controle-automatico).

A extensão adicionou previsão do próximo estado (velocidade, rotação, torque e consumo), rollouts de 10 s e comparação entre replay e uma política de acelerador ±5 pontos percentuais com demanda de referência e filtro de suporte. O fallback ocorreu em **52,71% a 100%** dos passos e as restrições não foram atendidas de forma suficiente. As diferenças de consumo são internas ao simulador aprendido, não economia real. Diem envolve tempo de abertura e pressão de injeção. Permanecem pendentes sua representação numérica, a disponibilidade dos comandos separados e a unidade/calibração de Taccm. Veja as tabelas atualizadas de [controle automático](APLICACOES_PREVISAO_CONSUMO.md#controle-automatico).

### Resultados preliminares das previsões

MAE para previsão de `Tau(t+10)` com o exemplo de previsão de torque (N·m):

| Ensaio de teste | Persistência | Modelo com sensores |
| --- | ---: | ---: |
| D1T1A | 279,1 | 272,2 |
| D1T1B | 282,5 | 306,1 |
| D1T2 | 237,0 | 318,1 |
| D2T1 | 249,5 | 304,4 |

O modelo só superou a persistência em D1T1A. Isso é um resultado útil para
o relatório: há variação entre ensaios e a previsão ainda não generaliza
de modo convincente. Investiguem janelas de atributos anteriores ao
instante atual e particularidades dos ensaios antes de ajustar modelos
mais complexos.

MAE para `Taccm(t+10)` com o exemplo de previsão de consumo, nas **mesmas linhas** em que
`Pi_6_log10` é válido:

| Ensaio de teste | Linhas | Persistência | Modelo Pi | Modelo com sensores |
| --- | ---: | ---: | ---: | ---: |
| D1T1A | 1.157 | 11,359 | 10,723 | 10,256 |
| D1T1B | 899 | 13,298 | 13,302 | 12,472 |
| D1T2 | 2.204 | 10,139 | 11,982 | 11,625 |
| D2T1 | 934 | 10,272 | 13,763 | 12,237 |

O conjunto Pi não melhorou a previsão de consumo nessa configuração.
Como o teste inclui apenas `Pi_6` positivo, para o qual o log é definido, esses
números **não** representam todos os instantes, especialmente paradas e
consumo zero. A comparação serve como resultado inicial, não como
conclusão sobre a utilidade de todos os grupos Pi.

### Classificação de severidade quando houver rótulos

Peçam a um responsável técnico ou a registros de manutenção uma definição observável: evento, janela temporal, critérios, classe e validação. Só então juntem os rótulos por `Experimento` e `Temp` e treinem um classificador. Comparem `DummyClassifier` e `RandomForestClassifier` com um ensaio inteiro deixado para teste, reportando matriz de confusão, precisão e recall da classe crítica e macro F1. Se alguma classe não aparecer no ensaio de teste, declarem a limitação em vez de calcular uma métrica enganosa.

É possível construir um **índice de carga operacional** por regra, por exemplo usando percentis de `Tau` e `n`, e estudar sua frequência. Isso seria uma definição de índice baseada em regras, não um rótulo independente de severidade. Treinar um modelo com `Tau` para reproduzir uma classe definida somente por limiares de `Tau` demonstraria a regra, não uma descoberta de falhas.

## Estrutura sugerida para o relatório

1. Descrever as perguntas, os quatro ensaios, aliases, unidades e dados ausentes.
2. Mostrar gráficos temporais de `Tau`, comandos, rotação e consumo.
3. Explicar pré-processamento, baselines, parâmetros e separação por ensaio.
4. Apresentar perfis de regimes, exemplos de anomalias e MAE de cada ensaio.
5. Discutir limitações: poucos ensaios, possíveis erros de sensores e ausência de falhas rotuladas.
