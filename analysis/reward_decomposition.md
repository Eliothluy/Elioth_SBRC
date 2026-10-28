# Decomposição do reward do artigo sobre dados reais


Gerado por `analysis/reward_decomposition.py` a partir de `/mnt/legitbet/elioth_sbrc/ns-o-ran-gym/examples/output/real_runs/baselines_dataset_d1d3.npz` (2079 transições, 8 valores distintos de `bs_on`). As componentes guardadas no dataset são as **normalizadas** (contrato de `nsoran.drl.offline`), pelo que a Eq. (2) pode ser verificada directamente. Modelo do ambiente: energia `watts`, custo de activação `dense` e normalização por célula.

Calibração quantílica ajustada nos runs reais (`reward_calibration_d1d3.npz`), amostras por componente. Para `es_on_cost` e `bs_on` a série global é a soma sobre as 7 células; a normalização D3 usa as séries `*_cell_*` e agrega pela média:

| componente | p0 | p25 | p50 | p75 | p100 |
|---|---|---|---|---|---|
| throughput | 0.000 | 46,880.000 | 59,204.000 | 73,683.000 | 130,258.000 |
| energy | 525.000 | 804.734 | 812.173 | 864.468 | 935.022 |
| rlf | 0.000 | 8.000 | 11.000 | 15.000 | 44.000 |
| es_on_cost | 0.000 | 0.007 | 0.150 | 0.903 | 6.300 |
| bs_on | 0.000 | 5.000 | 5.000 | 6.000 | 7.000 |

## Verificação da Eq. (2)


| agente | w1 | w2 | w3 | w4 | normalização | erro máximo de reconstrução |
|---|---|---|---|---|---|---|
| PPO-1 | 0.51 | 0.19 | 0.20 | 0.10 | normal | 4.089e-01 |
| PPO-2 | 0.40 | 0.40 | 0.10 | 0.10 | uniform | 4.768e-08 |
| PPO-3 | 0.20 | 0.40 | 0.20 | 0.20 | uniform | 2.683e-01 |
| PPO-4 | 0.40 | 0.32 | 0.18 | 0.10 | uniform | 1.096e-01 |
| PPO-5 | 0.45 | 0.20 | 0.25 | 0.10 | normal | 3.290e-01 |
| DQN | 0.40 | 0.40 | 0.10 | 0.10 | uniform | 4.768e-08 |

O reward armazenado é reproduzido por **PPO-2** com erro máximo de `4.77e-08` — a implementação é fiel à Eq. (2) com estes pesos. A decomposição seguinte usa esses pesos.

## Componentes


| componente | símbolo | sinal | peso | média normalizada | dp | min | max | contribuição média | % da massa negativa | corr(r, x) | corr dentro da política | corr(x, thr) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| throughput | w1 | + | 0.40 | 0.500 | 0.115 | 0.126 | 0.795 | 0.200 | — | -0.527 | -0.360 | 1.000 |
| energy | w2 | − | 0.40 | 0.500 | 0.114 | 0.123 | 0.854 | -0.200 | 40.0 | -0.712 | -0.646 | 0.781 |
| rlf | w3 | − | 0.10 | 0.500 | 0.097 | 0.197 | 0.688 | -0.050 | 10.0 | 0.208 | 0.152 | -0.019 |
| es_on_cost | w4 | − | 0.10 | 0.500 | 0.191 | 0.123 | 0.988 | -0.050 | 10.0 | -0.522 | -0.536 | 0.157 |
| bs_on | w2 | − | 0.40 | 0.500 | 0.278 | 0.007 | 0.921 | -0.200 | 40.0 | -0.953 | -0.932 | 0.648 |

O termo `bs_on` contribui com **-0.2000** por passo, isto é, **40.0%** de toda a massa negativa do reward; a energia contribui com 40.0% e o RLF com 10.0%. A correlação entre `bs_on` e o reward é -0.953, enquanto a do throughput é -0.527.

Como `w2` é usado duas vezes (energia e `bs_on`), o peso efectivo do eixo energético é `w2*(energy + bs_on)` = 0.40·(soma das duas componentes), superior ao `w2` nominal da Tabela I. Ver `mathematical-formulation.md`, R4.

Depois da normalização quantílica cada componente tem média ≈ 0,5; a contribuição **média** é portanto fixada pelos pesos, não pelos KPIs. O que distingue as políticas é a dispersão: `bs_on` tem o maior desvio-padrão (0.278), seguido de `es_on_cost` (0.191); as componentes de throughput, energia e RLF ficam em ≈ 0,10–0,12. É esta variância de `bs_on`, e não a energia por bit, que domina a ordenação do reward entre políticas.

A coluna `corr dentro da política` remove a média de cada política antes de correlacionar. Se o sinal agregado desaparecer (ou inverter) dentro das políticas, a correlação agregada era um efeito de composição entre políticas, não uma relação causal entre a KPI e o reward.

## Por política


| política | n | reward médio | thr | energia | RLF | es_on_cost | bs_on | contribuição thr | energia | RLF | activação | bs_on |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Always-On | 297 | -0.466 | 0.622 | 0.609 | 0.445 | 0.583 | 0.921 | 0.249 | -0.244 | -0.045 | -0.058 | -0.369 |
| Static | 792 | -0.273 | 0.498 | 0.502 | 0.532 | 0.465 | 0.429 | 0.199 | -0.201 | -0.053 | -0.047 | -0.172 |
| Dynamic | 693 | -0.286 | 0.502 | 0.494 | 0.515 | 0.493 | 0.471 | 0.201 | -0.198 | -0.052 | -0.049 | -0.189 |
| Random | 297 | -0.238 | 0.378 | 0.399 | 0.434 | 0.527 | 0.334 | 0.151 | -0.160 | -0.043 | -0.053 | -0.134 |

Todas as políticas que desligam células recebem uma penalização de `bs_on` menor do que o Always-On; é essa diferença, e não a energia por bit, que domina a ordenação do reward. O Random tem o pior RLF e o pior throughput, mas ainda assim reduz a penalização de `bs_on` em relação ao Always-On.

## Sensibilidade aos pesos da Tabela I


| agente | w1 | w2 | w3 | w4 | Always-On | Static | Dynamic | Random | melhor política | spread melhor−pior |
|---|---|---|---|---|---|---|---|---|---|---|
| PPO-1 | 0.51 | 0.19 | 0.20 | 0.10 | -0.121 | -0.076 | -0.080 | -0.086 | Static | 0.045 |
| PPO-2 | 0.40 | 0.40 | 0.10 | 0.10 | -0.466 | -0.273 | -0.286 | -0.238 | Random | 0.228 |
| PPO-3 | 0.20 | 0.40 | 0.20 | 0.20 | -0.693 | -0.472 | -0.487 | -0.410 | Random | 0.284 |
| PPO-4 | 0.40 | 0.32 | 0.18 | 0.10 | -0.379 | -0.241 | -0.250 | -0.214 | Random | 0.165 |
| PPO-5 | 0.45 | 0.20 | 0.25 | 0.10 | -0.196 | -0.142 | -0.145 | -0.138 | Random | 0.058 |
| DQN | 0.40 | 0.40 | 0.10 | 0.10 | -0.466 | -0.273 | -0.286 | -0.238 | Random | 0.228 |

Vencedor por conjunto de pesos: Static (PPO-1); Random (PPO-2, PPO-3, PPO-4, PPO-5, DQN). O Random — a pior política em throughput e RLF — ganha em 5 de 6 casos; o Static é escolhido por PPO-1, o conjunto com menor `w2` (0,19), que reduz o peso de `bs_on` e dá mais espaço ao throughput (o único termo positivo). O spread melhor−pior é máximo no PPO-3 (0.284) e mínimo no PPO-1 (0.045). Isto mostra que a Eq. (2), com esta normalização, **não codifica a restrição de QoS** que o artigo usa para avaliar os agentes: o reward e a métrica de avaliação (energia vs throughput vs RLF) apontam para políticas diferentes.

## Implicações para a reprodução


1. O reward offline é dominado por `bs_on`, uma variável de **estado** e não de desempenho; um agente pode aumentá-lo desligando células sem melhorar eficiência energética. Com os pesos da Tabela I, o Random — a pior política em throughput e RLF — é o óptimo em 5 de 6 casos; o PPO-1 é a excepção e prefere Static.
2. A energia usa o modelo EARTH `P_on = 130 + 94·x` W e `P_sleep = 75 W` (D1). A correlação `corr(thr, energy)` no dataset é 0.781: continua alta porque a carga `x` acompanha o throughput, mas o termo estático de 130 W impõe um piso e o sleep de 75 W limita a poupança.
3. A normalização quantílica por célula (D3) força média ≈ 0,5 em todas as componentes, pelo que a contribuição média de cada termo é fixada pelos pesos. A ordenação entre políticas vem da dispersão — sobretudo de `bs_on` (dp 0.278) — e não da magnitude média dos KPIs.
4. Qualquer comparação com a Tabela VI do artigo deve reportar também a intensidade energética (energia/throughput), caso contrário a poupança atribuída ao agente é indistinguível de tráfego não entregue.
5. O dataset usa a calibração quantílica ajustada nos runs reais (`reward_calibration_d1d3.npz`). Sem `--calibration-file`, as componentes ficam comprimidas pelas gamas de fallback e a decomposição deixa de ser comparável.
