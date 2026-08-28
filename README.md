# Visualização de séries temporais de veículos com pandas

Este projeto Python, gerenciado pelo UV, usa o pandas para ler a planilha
`Data` do arquivo `data/1oDia_Teste2.xlsx` e gerar um gráfico de um intervalo
selecionado de dois atributos:

- `TachographVehicleSpeed`
- `EngineSpeed`

Por padrão, são utilizadas as linhas de 0 a 499. Como os atributos possuem
escalas diferentes, o gráfico utiliza um eixo y separado para cada série. A
imagem PNG resultante é salva em `outputs/speed_and_engine_speed.png`.

## Instalar o UV

O UV gerencia o Python 3.14, o ambiente virtual e todas as dependências do
projeto. Não é necessário instalar o Python separadamente nem criar ou ativar
um ambiente virtual manualmente.

### Linux

Abra um terminal e execute:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Reinicie o terminal e verifique a instalação:

```bash
uv --version
```

### macOS

Use o mesmo instalador independente utilizado no Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Como alternativa, use o Homebrew:

```bash
brew install uv
```

Reinicie o terminal e verifique a instalação com `uv --version`.

### Windows

Abra o PowerShell e execute:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Reinicie o PowerShell e verifique a instalação:

```powershell
uv --version
```

Documentação oficial de instalação:
<https://docs.astral.sh/uv/getting-started/installation/>

## Executar o projeto

Os comandos a seguir são os mesmos no Linux, macOS, PowerShell e Prompt de
Comando do Windows. A partir do diretório do projeto, sincronize o ambiente:

```console
uv sync
```

Gere o gráfico padrão e abra a janela interativa do Matplotlib:

```console
uv run python plot_timeseries.py
```

O UV utiliza automaticamente o Python 3.14 definido em `.python-version` e
instala as dependências declaradas em `pyproject.toml`. Caso essa versão do
Python ainda não esteja disponível, o UV poderá baixá-la. O projeto usa o
backend QtAgg fornecido pelo PyQt6 para abrir a janela do gráfico no Linux,
Windows e macOS.

O gráfico também será salvo em `outputs/speed_and_engine_speed.png`. Em um
servidor ou ambiente sem interface gráfica, use `--no-show` para apenas salvar
a imagem sem abrir uma janela:

```console
uv run python plot_timeseries.py --no-show
```

## Personalizar o gráfico

Para representar outro intervalo de linhas ou duas colunas diferentes da mesma
planilha:

```console
uv run python plot_timeseries.py \
  --start 500 \
  --end 1000 \
  --first EngineFuelRate \
  --second AcceleratorPedalAPPosition1 \
  --output outputs/fuel_and_accelerator.png
```

No PowerShell do Windows, insira o comando em uma única linha ou substitua cada
barra invertida (`\`) no final das linhas por um acento grave (`` ` ``).

Também é possível selecionar outro arquivo ou outra planilha:

```console
uv run python plot_timeseries.py --input data/2oDia_Teste1.xlsx --sheet Data
```

Para visualizar todas as opções:

```console
uv run python plot_timeseries.py --help
```

## Executar os testes

```console
uv run pytest
```
