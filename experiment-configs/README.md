# experiment-configs/

Manifestos versionáveis dos experimentos propostos em `reproduction-plan.md`.
São **configurações de intenção**, não resultados: nenhum run descrito aqui foi
executado além dos 24 baselines já existentes.

| Arquivo | Conteúdo | Status |
|---|---|---|
| `baselines_5seeds.json` | 8 políticas × 5 seeds = 40 runs | 21/40 já existem |
| `dataset_plan.json` | 400 runs de comportamento para treino offline | a executar |
| `training_matrix.json` | 18 treinos + 24 de ablação | a executar |
| `eval_matrix.json` | 90 avaliações live + 24 de ablação | a executar |
| `cost_model.json` | custo medido e projeção por tier | referência |
| `decisions.md` | D1–D5 registrados na aprovação | **a criar** |

## Regras

1. **Não editar** `src/environments/scenario_configurations/paper_es_use_case.json`.
   O hash está pinado em `baselines_5seeds.json`
   (`3def717c…83f2e8`); se mudar, os resultados deixam de ser comparáveis.
2. Todo run deve gravar `seed`, comando e timestamp no `manifest.json` do
   diretório de saída. Sem isso, o run não entra na análise.
3. Nenhuma configuração usa `--synthetic`. Dados sintéticos servem só para
   smoke test de pipeline.
4. Avaliação de política sempre com `--ns3-path`; `--env dataset` é replay.

## Ordem de execução

```
Fase 0  sanidade            → 1 run
Fase 1  calibração          → 0 runs (reusa 21)
Fase 2  baselines           → 19 runs novos
Fase 3  dataset             → 400 runs
Fase 4  treino              → 42 treinos (CPU, sem ns-3)
Fase 5  avaliação           → 114 runs
Fase 6  análise             → 0 runs
```

Total Tier M: **554 runs de simulação**, ~720 h de CPU, ~130 GB.
