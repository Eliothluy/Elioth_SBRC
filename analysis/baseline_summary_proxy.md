# Baselines recompostos a partir das execuções reais em ns-3

Gerado por `analysis/recompute_baselines.py`. Cada linha vem de uma pasta de execução (`database.db` + `bsState.txt`) reprocessada por `examples/compare_runs.py::run_metrics` (janela integral; as estatísticas usam as 99 indicações do cenário de 10 s). Nenhum número desta tabela foi copiado de tabela agregada.

Manifesto: `/mnt/legitbet/elioth_sbrc/ns-o-ran-gym/examples/output/real_runs/manifest.json` — 24 jobs, 21 completos e comparáveis, 3 abortados/excluídos.

Modelo proxy (ablação da Eq. (1)): energia = `TB.TotNbrDl.1 * 1.0`, custo de ativação `dense` e normalização por célula (D3).

## Por política

| política | runs | throughput (média ± IC95) | energia (média ± IC95) | RLF médio | RLF máx | BS ligadas | Δthr % | Δenergia % | intensidade E/thr vs Always-On |
|---|---|---|---|---|---|---|---|---|---|
| always_on | 3 | 76,686 ± 39,613 | 747.5 ± 331.0 | 7.20 | 18 | 7.00 | 0.00 | 0.00 | 1.000 |
| static | 8 | 62,196 ± 7,733 | 642.4 ± 61.5 | 12.43 | 26 | 5.12 | -18.90 | -14.06 | 1.060 |
| dynamic | 7 | 59,769 ± 9,040 | 628.5 ± 81.5 | 11.52 | 27 | 5.29 | -22.06 | -15.91 | 1.079 |
| random | 3 | 31,407 ± 21,291 | 358.7 ± 130.3 | 16.30 | 44 | 3.90 | -59.04 | -52.01 | 1.172 |

`intensidade E/thr` = energia média por unidade de throughput médio, normalizada pelo valor do Always-On. Valores > 1 significam que a política gasta **mais** energia por bit entregue: a poupança absoluta vem de transportar menos tráfego, não de operar a rede de forma mais eficiente.

## Por tupla (Static/Dynamic)

As tuplas do artigo têm efeitos muito diferentes; a média por política esconde essa dispersão.

| política | tupla (N_On,P1,N_On,P2,N_Off) | runs | throughput | energia | RLF | BS ligadas | Δthr % | Δenergia % | E/thr vs Always-On |
|---|---|---|---|---|---|---|---|---|---|
| static | 2-2-0-3 | 2 | 54,954 | 565.4 | 17.57 | 4.00 | -28.34 | -24.36 | 1.056 |
| static | 3-2-0-2 | 3 | 62,027 | 647.2 | 11.61 | 5.00 | -19.12 | -13.42 | 1.070 |
| static | 4-2-0-1 | 3 | 67,192 | 688.9 | 9.83 | 6.00 | -12.38 | -7.84 | 1.052 |
| dynamic | 2-2-0-3 | 1 | 44,931 | 517.6 | 16.82 | 4.00 | -41.41 | -30.76 | 1.182 |
| dynamic | 3-2-0-2 | 3 | 57,417 | 589.0 | 12.41 | 5.00 | -25.13 | -21.20 | 1.052 |
| dynamic | 4-2-0-1 | 3 | 67,067 | 705.1 | 8.88 | 6.00 | -12.54 | -5.67 | 1.079 |

## Execuções abortadas

| run | política | tupla | seed | sim time | indicações | nota |
|---|---|---|---|---|---|---|
| dynamic_2_2_0_3_seed555 | dynamic | 2-2-0-3 | 555 | 1.100 | 11 | crashed at 1.1 s, 11 of 99 indications |
| dynamic_2_2_0_3_seed556 | dynamic | 2-2-0-3 | 556 | 7.600 | 76 | crashed at 7.6 s, 76 of 99 indications |
| static_2_2_0_3_seed556 | static | 2-2-0-3 | 556 | 7.600 | 76 | crashed at 7.6 s, 76 of 99 indications |

As execuções abortadas ficam fora da agregação principal. As execuções `dynamic_2_2_0_3_seed556` e `static_2_2_0_3_seed556` têm impressão digital idêntica (mesmo throughput, energia, RLF e bs_on): não são amostras independentes.

## Leitura

- O Always-On tem enorme variância entre seeds (ver `baseline_per_run.csv`), o que fixa um piso de incerteza para qualquer comparação de políticas.
- Nenhuma política preserva throughput; a ordenação por energia é quase a ordenação por throughput perdido. A coluna de intensidade mostra que a eficiência por bit não melhora de forma sistemática.
- As tuplas 4-2-0-1 são as menos agressivas e as 2-2-0-3 as mais agressivas, como esperado pela definição (N_On, N_Off).
- O custo de activação médio das políticas que desligam células é próximo do das Always-On porque a métrica `0.9**(0.01*T_on_ms)` recompensa tempo ligado, não transições OFF→ON (ver `mathematical-formulation.md`, R2).
