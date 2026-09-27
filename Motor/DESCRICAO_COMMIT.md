# Descrição das alterações: análise de consumo e operação do motor

## Título sugerido para o commit

`feat: adiciona experimentos de ML para consumo e operação do motor`

Este documento descreve o conjunto de alterações preparado na branch `analise/consumo-motor-resultados`. O conteúdo efetivo do commit dependerá dos arquivos selecionados para inclusão.

## Motivação e objetivo

O projeto da disciplina de Machine Learning investiga dados de sensores de um ônibus para encontrar padrões de operação e avaliar possíveis indicadores de severidade. Esta entrega organiza os dados, documenta abordagens supervisionadas e não supervisionadas e executa quatro aplicações de previsão de consumo e análise operacional.

As perguntas principais são:

1. Podemos antecipar períodos de consumo elevado para orientar o motorista?
2. Podemos prever o consumo acumulado de um intervalo para apoiar planejamento energético?
3. Existem regimes de operação diferentes nos sinais, especialmente torque (`Tau`), rotação e acelerador?
4. Um modelo de dinâmica permite comparar ações de acelerador em uma simulação com restrições?

Não há rótulos confirmados de falha ou severidade. Portanto, a entrega apresenta padrões e avaliações de previsão, sem atribuir automaticamente severidade aos grupos encontrados.

## O que entra nesta alteração

A entrega contém somente arquivos novos em relação à base `main`. O `README.md`, o `.gitignore` da raiz e os demais arquivos preexistentes são preservados integralmente.

| Arquivos ou diretório | Conteúdo e finalidade |
| --- | --- |
| [README do motor](README.md) | Cria um ponto de entrada para a análise, preservando o README preexistente da raiz. |
| [.gitignore do motor](.gitignore) e [docs/.gitignore](../docs/.gitignore) | Acrescentam regras locais para checkpoints, dados duplicados, gráficos Pi regeneráveis e planos internos, preservando o .gitignore da raiz. |
| [SUGESTOES_MACHINE_LEARNING.md](SUGESTOES_MACHINE_LEARNING.md) | Organiza propostas supervisionadas e não supervisionadas com os sensores e grupos Pi disponíveis; distingue sugestões de experimentos executados. |
| [APLICACOES_PREVISAO_CONSUMO.md](APLICACOES_PREVISAO_CONSUMO.md) | Explica objetivos, métodos, resultados por ensaio, limitações, interpretação e reprodução das quatro aplicações. |
| [Main_Motor.py](Main_Motor.py) e `Auxi/` | Disponibiliza os scripts extraídos do material original, usados para gerar bases, grupos Pi, diagnósticos e gráficos. |
| `Dados/` | Preserva os dados de entrada, a planilha de referência, os parâmetros dos sensores e os CSVs derivados da análise dimensional. |
| `notebooks/` | Acrescenta quatro notebooks executados, funções compartilhadas, executor e dependências. |
| `resultados_consumo/` | Preserva métricas, previsões, agrupamentos, cenários, gráficos e informações de rastreabilidade das execuções. |
| [test_consumo_temporal.py](../tests/test_consumo_temporal.py) | Verifica janelas temporais, alinhamento dos alvos, séries irregulares e consistência dos cenários de acelerador. |
| [test_consumo_melhorias.py](../tests/test_consumo_melhorias.py) | Verifica seleção de limiar, agrupamento de avisos, comparação de históricos, calibração, restrições e alinhamento da dinâmica. |
| [DESCRICAO_COMMIT.md](DESCRICAO_COMMIT.md) | Reúne o contexto e o escopo da entrega para revisão do commit. |

Os scripts originais e os experimentos novos ficam disponíveis juntos para permitir rastrear os resultados até os dados. O arquivo `Motor.zip` já estava versionado antes desta alteração.

## Dados e metodologia

A base contém **9.339 amostras em quatro ensaios**: D1T1A, D1T1B, D1T2 e D2T1. Os blocos originais são empilhados preservando o identificador do ensaio e o tempo. Os Pi são associados às leituras por essas duas chaves.

As entradas incluem torque real (`Tau`), velocidade, rotação, acelerador, freio, marcha, temperaturas, consumo atual e atributos históricos. Os dados brutos são preservados.

Nas avaliações supervisionadas, um ensaio inteiro fica reservado para teste enquanto os demais servem para treino. As entradas usam somente o instante atual e o passado; os alvos são posteriores. As extensões selecionam limiares e configurações em validação interna, sem usar o ensaio externo nesses ajustes. Persistência e outras referências simples permitem avaliar se o modelo acrescenta valor.

Os agrupamentos são exploratórios, ajustados sobre a base disponível. As quatro coletas já foram exploradas durante o projeto; a avaliação entre ensaios não substitui uma confirmação em novas coletas independentes.

## Experimentos e resultados

### 1. Orientação ao motorista — supervisionado

[Notebook 01](notebooks/01_orientacao_motorista.ipynb).

Uma Random Forest classifica se a média de consumo dos próximos 10 segundos será elevada. O limite de consumo elevado é definido com os dados de treino. A extensão seleciona o limiar de probabilidade na validação interna e agrupa avisos consecutivos.

- Falsos positivos por janela: **113 → 79**.
- Falsos negativos por janela: **35 → 50**.
- Avisos falsos agrupados: **47 → 35**.

**Interpretação:** houve redução dos alertas falsos, acompanhada de mais eventos não detectados. A escolha do limiar depende do custo desses dois tipos de erro.

### 2. Planejamento energético — supervisionado

[Notebook 02](notebooks/02_planejamento_energetico.ipynb).

Um regressor estima a integral de consumo dos próximos 30 segundos. A extensão compara históricos de 10 e 30 segundos, presença ou ausência de `Pi_6` e uma correção aditiva de viés calculada com previsões da validação interna.

**Resultado:** a calibração piorou o erro percentual absoluto do total nos **quatro ensaios externos**. Ela não demonstrou ser uma correção transferível entre ensaios. A qualidade de previsão por janela e o erro no total são avaliados separadamente.

### 3. Análise de operação — não supervisionado

[Notebook 03](notebooks/03_analise_operacao.ipynb).

KMeans agrupa **310 janelas completas de 30 segundos**, usando sensores e, em uma segunda análise, uma representação com Pi. O consumo não entra no agrupamento; é usado posteriormente para descrever os grupos.

- Dois grupos foram selecionados entre as alternativas avaliadas.
- Silhouette: **0,296** com sensores e **0,283** com Pi.
- Consumo médio dos grupos de sensores: **4,37 U** e **15,65 U**.

**Interpretação:** há perfis operacionais com consumos diferentes. Isso não comprova desperdício, falha ou severidade; diferenças de carga e demanda podem explicar parte da separação.

### 4. Controle automático — previsão e simulação exploratória

[Notebook 04](notebooks/04_controle_automatico.ipynb).

A análise inicial examina a sensibilidade do preditor a alterações hipotéticas do acelerador. A extensão aprende a dinâmica de velocidade, rotação, torque e consumo com um passo de um segundo e simula trajetórias de 10 segundos.

A política compara acelerador nominal e variações de ±5 pontos percentuais. As alternativas passam por um filtro de proximidade aos dados de treino e por restrições de velocidade e torque. A comparação usa a mesma trajetória de referência e agendas exógenas conhecidas para as duas políticas.

- Diferenças de consumo internas ao simulador: aproximadamente **−2,74% a 0%** frente ao replay.
- Retorno ao comando nominal por falta de candidato elegível (*fallback*): **52,71% a 100%** dos passos.
- Houve violações das restrições; o fallback não garante atendimento à demanda.

**Interpretação:** esses números não representam economia medida no ônibus. O experimento não valida controle real de injeção nem identifica efeitos causais de intervenções.

## Unidades e esclarecimento de Diem

A unidade/calibração de `Taccm` permanece sem confirmação. Por isso, os relatórios usam **U** para a taxa numérica registrada e **U·s** para sua integral, sem conversão para litros, dinheiro ou autonomia.

**Diem envolve tempo de abertura da injeção + pressão de injeção**. Esse “+” descreve a participação das duas grandezas, sem estabelecer uma soma numérica entre unidades diferentes.

O arquivo contém uma única coluna escalar `Diem` por ensaio, rotulada como m³/s, e a variável está ausente em dois ensaios. Ainda falta confirmar a relação entre esse valor, duração de abertura e pressão, bem como se ele representa medição, comando ou cálculo.

- Os modelos executados não usam `Diem` nem `Pi_5` como entrada.
- Os documentos propõem análises futuras com `Diem`, identificadas como ainda não executadas.
- `Pi_5` só é adimensional sob a hipótese documental de `Diem` com dimensão de vazão. Uma mudança nessa dimensão exige revisar os metadados e recalcular os Pi.
- A simulação altera acelerador; não controla separadamente pressão e duração de abertura.

## Evidências e reprodução

Os notebooks contêm tabelas e gráficos das execuções. Os CSVs em `resultados_consumo/` permitem conferir métricas e previsões; arquivos `ambiente.json` registram versões, sementes e hashes de entradas. As extensões também registram hashes dos auxiliares usados. O notebook de controle foi reexecutado após o esclarecimento sobre `Diem`.

Os testes adicionados cobrem propriedades temporais e decisões dos algoritmos. Eles não demonstram validade física dos sensores nem eficácia de um controlador real.

A execução registrada utilizou **Python 3.12.3** em ambiente separado. O requisito Python 3.14 do projeto principal foi preservado. A partir da raiz, em um ambiente com as dependências adequadas:

```bash
python -m pip install -r Motor/notebooks/requirements.txt
python Motor/notebooks/executar_notebooks.py
```

Para regenerar os CSVs Pi e os gráficos exploratórios, execute `Main_Motor.py` com o diretório de trabalho em `Motor/`. Os CSVs necessários aos notebooks estão incluídos na entrega. O executor de notebooks requer que o ambiente permita os sockets locais do Jupyter.

## Critério de seleção dos arquivos

Foram mantidos dados de entrada, código, notebooks executados, testes e resultados que permitem compreender ou auditar as conclusões. As previsões individuais e os passos da simulação foram preservados porque permitem conferir os totais apresentados.

As regras preexistentes da raiz continuam excluindo caches. Os novos arquivos `Motor/.gitignore` e `docs/.gitignore` excluem checkpoints, planos internos, a pasta duplicada `Data_set limpo/` e os gráficos exploratórios Pi regeneráveis. As figuras referenciadas no relatório permanecem disponíveis para versionamento. Os arquivos ignorados não foram apagados do disco.

## Pontos para revisão técnica

1. Conferir os alvos e a separação entre treino, validação e teste.
2. Avaliar o compromisso entre falsos positivos e eventos não detectados.
3. Investigar a variação entre ensaios e o fracasso da calibração acumulada.
4. Interpretar os agrupamentos considerando carga, trajeto e demanda de torque.
5. Confirmar unidades e o mapeamento de `Diem` antes de conclusões físicas adicionais.
6. Definir, com critérios externos aos agrupamentos, o que significaria severidade neste projeto.

## Síntese

A entrega transforma os dados e as ideias iniciais em experimentos reproduzíveis de ML, com resultados favoráveis e desfavoráveis documentados. Há evidências exploratórias de padrões operacionais e de utilidade para previsão, mas persistem limitações de generalização, calibração e interpretação física. Não foi demonstrado um indicador validado de severidade nem um sistema pronto para controle automático do veículo.
