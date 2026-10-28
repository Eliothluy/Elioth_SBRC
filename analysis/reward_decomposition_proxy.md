# Decomposição do reward do artigo sobre dados reais


Gerado por `analysis/reward_decomposition.py` a partir de `ns-o-ran-gym/examples/output/real_runs/baselines_dataset.npz` (2079 transições, 8 valores distintos de `bs_on`). As componentes guardadas no dataset são as **normalizadas** (contrato de `nsoran.drl.offline`), pelo que a Eq. (2) pode ser verificada directamente.

## Verificação da Eq. (2)


| agente | w1 | w2 | w3 | w4 | normalização | erro máximo de reconstrução |
|---|---|---|---|---|---|---|
| PPO-1 | 0.51 | 0.19 | 0.20 | 0.10 | normal | 2.134e-01 |
| PPO-2 | 0.40 | 0.40 | 0.10 | 0.10 | uniform | 2.831e-08 |
| PPO-3 | 0.20 | 0.40 | 0.20 | 0.20 | uniform | 1.316e-01 |
| PPO-4 | 0.40 | 0.32 | 0.18 | 0.10 | uniform | 8.035e-02 |
| PPO-5 | 0.45 | 0.20 | 0.25 | 0.10 | normal | 2.020e-01 |
| DQN | 0.40 | 0.40 | 0.10 | 0.10 | uniform | 2.831e-08 |

O reward armazenado é reproduzido por **PPO-2** com erro máximo de `2.83e-08` — a implementação é fiel à Eq. (2) com estes pesos. A decomposição seguinte usa esses pesos.

## Componentes


| componente | símbolo | sinal | peso | média normalizada | dp | min | max | contribuição média | % da massa negativa | corr(r, x) | corr dentro da política | corr(x, thr) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| throughput | w1 | + | 0.40 | 0.011 | 0.002 | 0.005 | 0.018 | 0.004 | — | -0.438 | -0.236 | 1.000 |
| energy | w2 | − | 0.40 | 0.011 | 0.002 | 0.005 | 0.017 | -0.004 | 1.3 | -0.427 | -0.233 | 0.893 |
| rlf | w3 | − | 0.10 | 0.192 | 0.100 | 0.005 | 0.696 | -0.019 | 5.8 | 0.712 | 0.653 | -0.497 |
| es_on_cost | w4 | − | 0.10 | 0.108 | 0.168 | 0.005 | 0.896 | -0.011 | 3.2 | -0.273 | -0.429 | -0.224 |
| bs_on | w2 | − | 0.40 | 0.751 | 0.190 | 0.005 | 0.995 | -0.300 | 89.7 | -0.978 | -0.970 | 0.531 |

O termo `bs_on` contribui com **-0.3002** por passo, isto é, **89.7%** de toda a massa negativa do reward; a energia contribui com 1.3% e o RLF com 5.8%. A correlação entre `bs_on` e o reward é -0.978, enquanto a do throughput é -0.438.

Como `w2` é usado duas vezes (energia e `bs_on`), o peso efectivo do eixo energético é `w2*(energy + bs_on)` = 0.40·(soma das duas componentes), superior ao `w2` nominal da Tabela I. Ver `mathematical-formulation.md`, R4.

A coluna `corr dentro da política` remove a média de cada política antes de correlacionar. Se o sinal agregado desaparecer (ou inverter) dentro das políticas, a correlação agregada era um efeito de composição entre políticas, não uma relação causal entre a KPI e o reward.

## Por política


| política | n | reward médio | thr | energia | RLF | es_on_cost | bs_on | contribuição thr | energia | RLF | activação | bs_on |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Always-On | 297 | -0.420 | 0.013 | 0.012 | 0.118 | 0.104 | 0.995 | 0.005 | -0.005 | -0.012 | -0.010 | -0.398 |
| Static | 792 | -0.320 | 0.011 | 0.011 | 0.200 | 0.080 | 0.730 | 0.004 | -0.005 | -0.020 | -0.008 | -0.292 |
| Dynamic | 693 | -0.330 | 0.011 | 0.011 | 0.186 | 0.100 | 0.753 | 0.004 | -0.004 | -0.019 | -0.010 | -0.301 |
| Random | 297 | -0.270 | 0.008 | 0.009 | 0.261 | 0.207 | 0.556 | 0.003 | -0.003 | -0.026 | -0.021 | -0.223 |

Todas as políticas que desligam células recebem uma penalização de `bs_on` menor do que o Always-On; é essa diferença, e não a energia por bit, que domina a ordenação do reward. O Random tem o pior RLF e o pior throughput, mas ainda assim reduz a penalização de `bs_on` em relação ao Always-On.

## Sensibilidade aos pesos da Tabela I


| agente | w1 | w2 | w3 | w4 | Always-On | Static | Dynamic | Random | melhor política | spread melhor−pior |
|---|---|---|---|---|---|---|---|---|---|---|
| PPO-1 | 0.51 | 0.19 | 0.20 | 0.10 | -0.219 | -0.183 | -0.187 | -0.176 | Random | 0.043 |
| PPO-2 | 0.40 | 0.40 | 0.10 | 0.10 | -0.420 | -0.320 | -0.330 | -0.270 | Random | 0.151 |
| PPO-3 | 0.20 | 0.40 | 0.20 | 0.20 | -0.445 | -0.350 | -0.361 | -0.318 | Random | 0.127 |
| PPO-4 | 0.40 | 0.32 | 0.18 | 0.10 | -0.349 | -0.277 | -0.284 | -0.245 | Random | 0.104 |
| PPO-5 | 0.45 | 0.20 | 0.25 | 0.10 | -0.236 | -0.201 | -0.204 | -0.195 | Random | 0.040 |
| DQN | 0.40 | 0.40 | 0.10 | 0.10 | -0.420 | -0.320 | -0.330 | -0.270 | Random | 0.151 |

Em **todos** os seis conjuntos de pesos da Tabela I o Random obtém o melhor reward médio, apesar de ter o pior throughput (–59% vs Always-On) e o pior RLF (16.3 vs 7.2 UEs por indicação). A razão é aritmética: `bs_on` pesa `w2` e domina a massa negativa, enquanto o RLF pesa apenas `w3` ≤ 0.25. O spread melhor−pior é máximo no PPO-2/DQN (0.151) e mínimo no PPO-5 (0.040). Isto mostra que a Eq. (2), com esta normalização, **não codifica a restrição de QoS** que o artigo usa para avaliar os agentes: o reward e a métrica de avaliação (energia vs throughput vs RLF) apontam para políticas diferentes.

## Implicações para a reprodução


1. O reward offline é dominado por `bs_on`, uma variável de **estado** e não de desempenho; um agente pode aumentá-lo desligando células sem melhorar eficiência energética. Com os pesos da Tabela I, o Random — a pior política em throughput e RLF — é o óptimo do reward em todos os seis casos.
2. A energia é o proxy `TB.TotNbrDl.1 * p_tx` (sem potência estática), pelo que `energy` e `throughput` são quase colineares; a correlação `corr(thr, energy)` no dataset é 0.893.
3. Qualquer comparação com a Tabela VI do artigo deve reportar também a intensidade energética (energia/throughput), caso contrário a poupança atribuída ao agente é indistinguível de tráfego não entregue.
4. A normalização usada no dataset é a de fallback (não calibrada nos runs reais); `examples/fit_reward_calibration.py` deve ser executado antes de treinar para que as componentes não fiquem comprimidas em ≈0.005.
