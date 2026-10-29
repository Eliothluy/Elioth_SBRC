# Baselines recompostos a partir das execuções reais em ns-3

Gerado por `analysis/recompute_baselines.py`. Cada linha vem de uma pasta de execução (`database.db` + `bsState.txt`) reprocessada por `examples/compare_runs.py::run_metrics` (janela integral; as estatísticas usam as 99 indicações do cenário de 10 s). Nenhum número desta tabela foi copiado de tabela agregada.

Manifesto: `/mnt/legitbet/elioth_sbrc/ns-o-ran-gym/examples/output/real_runs/manifest.json` — 40 jobs, 34 completos e comparáveis, 6 abortados/excluídos.

Modelo D1 (principal): potência EARTH macro por setor (P_0=130.0 W, P_tx,max=20.0 W, Δp=4.7, N=1.0, P_sleep=75.0 W; P_on=130+94x W), custo de ativação `dense` (D2) e normalização por célula (D3; afeta o reward, não a energia/throughput acima).

## Por política

| política | runs | throughput (média ± IC95) | energia (média ± IC95) | RLF médio | RLF máx | BS ligadas | Δthr % | Δenergia % | intensidade E/thr vs Always-On | latência PDCP p95 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| always_on | 5 | 74,732 ± 19,173 | 919.3 ± 1.5 | 7.59 | 18 | 7.00 | 0.00 | 0.00 | 1.000 | 202.9 |
| static | 12 | 58,807 ± 7,853 | 803.3 ± 28.1 | 13.14 | 26 | 4.92 | -21.31 | -12.62 | 1.110 | 337.2 |
| dynamic | 12 | 57,812 ± 6,855 | 812.3 ± 28.3 | 12.31 | 27 | 5.08 | -22.64 | -11.64 | 1.142 | 391.1 |
| random | 5 | 28,654 ± 8,876 | 722.6 ± 40.1 | 18.79 | 48 | 3.50 | -61.66 | -21.39 | 2.050 | 817.3 |

`intensidade E/thr` = energia média por unidade de throughput médio, normalizada pelo valor do Always-On. Valores > 1 significam que a política gasta **mais** energia por bit entregue: a poupança absoluta vem de transportar menos tráfego, não de operar a rede de forma mais eficiente.

Com o modelo D1 a célula ligada consome `130 + 94·x` W (x = PRB/139) e a desligada 75 W; desligar uma célula poupa no máximo 66,5% (carga máxima) e no mínimo 42,3% (carga zero). Por isso a poupança de energia é menor que a perda de throughput no Random, embora a intensidade E/thr ainda piore em todas as políticas. O artigo relata +0,7%/+4,3% de poupança para Static/Dynamic com −23,5%/−27,6% de throughput; esses valores não são reproduzíveis com a Eq. (1) literal nem com o modelo EARTH, e ficam registados como ambiguidade (ver `mathematical-formulation.md`).

## Por tupla (Static/Dynamic)

As tuplas do artigo têm efeitos muito diferentes; a média por política esconde essa dispersão.

| política | tupla (N_On,P1,N_On,P2,N_Off) | runs | throughput | energia | RLF | BS ligadas | Δthr % | Δenergia % | E/thr vs Always-On | p95 lat (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| static | 2-2-0-3 | 4 | 51,294 | 751.8 | 17.31 | 4.00 | -31.36 | -18.22 | 1.191 | 297.9 |
| static | 3-2-0-2 | 5 | 59,786 | 808.5 | 11.79 | 5.00 | -20.00 | -12.05 | 1.099 | 399.4 |
| static | 4-2-0-1 | 3 | 67,192 | 863.3 | 9.83 | 6.00 | -10.09 | -6.09 | 1.044 | 285.8 |
| dynamic | 2-2-0-3 | 3 | 48,639 | 751.5 | 16.75 | 4.00 | -34.92 | -18.25 | 1.256 | 320.3 |
| dynamic | 3-2-0-2 | 5 | 56,894 | 807.6 | 12.30 | 5.00 | -23.87 | -12.15 | 1.154 | 400.4 |
| dynamic | 4-2-0-1 | 4 | 65,839 | 863.9 | 8.99 | 6.00 | -11.90 | -6.03 | 1.067 | 432.6 |

## Execuções abortadas

| run | política | tupla | seed | sim time | indicações | nota |
|---|---|---|---|---|---|---|
| dynamic_2_2_0_3_seed555 | dynamic | 2-2-0-3 | 555 | 1.100 | 11 | crashed at 1.1 s, 11 of 99 indications |
| dynamic_2_2_0_3_seed556 | dynamic | 2-2-0-3 | 556 | 7.600 | 76 | crashed at 7.6 s, 76 of 99 indications |
| dynamic_4_2_0_1_seed558 | dynamic | 4-2-0-1 | 558 | 9.900 | 99 | crashed at 9.9 s, 99 of 99 indications |
| static_2_2_0_3_seed556 | static | 2-2-0-3 | 556 | 7.600 | 76 | crashed at 7.6 s, 76 of 99 indications |
| static_4_2_0_1_seed558 | static | 4-2-0-1 | 558 | 9.900 | 99 | crashed at 9.9 s, 99 of 99 indications |
| static_4_2_0_1_seed559 | static | 4-2-0-1 | 559 | 9.900 | 99 | crashed at 9.9 s, 99 of 99 indications |

As execuções abortadas ficam fora da agregação principal. As execuções `dynamic_2_2_0_3_seed556` e `static_2_2_0_3_seed556` têm impressão digital idêntica (mesmo throughput, energia, RLF e bs_on): não são amostras independentes.

## Leitura

- O Always-On tem enorme variância entre seeds (ver `baseline_per_run.csv`), o que fixa um piso de incerteza para qualquer comparação de políticas.
- Nenhuma política preserva throughput; a ordenação por energia é quase a ordenação por throughput perdido. A coluna de intensidade mostra que a eficiência por bit não melhora de forma sistemática.
- As tuplas 4-2-0-1 são as menos agressivas e as 2-2-0-3 as mais agressivas, como esperado pela definição (N_On, N_Off).
- O custo de activação médio das políticas que desligam células é próximo do das Always-On porque a métrica `0.9**(0.01*T_on_ms)` recompensa tempo ligado, não transições OFF→ON (ver `mathematical-formulation.md`, R2).
