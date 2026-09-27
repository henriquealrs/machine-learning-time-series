# Análise de consumo e operação do motor

## Documentação

- [Aplicações, objetivos e resultados](APLICACOES_PREVISAO_CONSUMO.md): métodos, métricas, limitações e reprodução dos quatro experimentos.
- [Sugestões de machine learning](SUGESTOES_MACHINE_LEARNING.md): propostas supervisionadas e não supervisionadas, com indicação do que já foi executado.
- [Descrição das alterações](DESCRICAO_COMMIT.md): escopo da entrega e pontos para revisão técnica.

## Conteúdo

- `Dados/`: dados de entrada, parâmetros e bases Pi.
- `Main_Motor.py` e `Auxi/`: geração das bases derivadas e gráficos Pi.
- `notebooks/`: quatro notebooks executados, auxiliares, executor e dependências.
- `resultados_consumo/`: métricas, previsões, figuras e registros das execuções.

As instruções de reprodução estão no relatório de aplicações. Os arquivos preexistentes do repositório foram preservados; esta entrega acrescenta arquivos novos. As regras específicas de exclusão estão em [.gitignore](.gitignore) e [docs/.gitignore](../docs/.gitignore).
