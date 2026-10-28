# Registro de decisões — D1 a D5

**Data:** 2026-10-22
**Aprovação:** o usuário autorizou explicitamente "implementar tudo" (D1–D5 e o
escopo de código), após a apresentação das Fases A/B/F e do plano de reprodução.
**Alcance da autorização:** alterar `ns-o-ran-gym` (ambiente, DRL, exemplos e
scripts), criar artefatos em `experiment-configs/` e `analysis/`, e lançar a
campanha de simulação. `ns-3-mmwave-oran` só será tocado se um KPM necessário
estiver ausente — o que, na inspeção feita, não é o caso.

Este arquivo é o registro vinculante: qualquer resultado produzido a partir
daqui usa estas definições. Desvios exigem uma nova entrada datada.

---

## D1 — Modelo de energia em watts

**Escolha:** (b) `P = P_0 + Δp·P_max·x`, com `P_sleep` quando a célula está OFF.
O proxy `TB.TotNbrDl.1 × 1.0` permanece disponível como ablation.

**Modelo (EARTH, macro, por sector, N = 1 antena):**

```
x        = min(prb_used / 139, 1)          # carga de PRB no período de 100 ms
P_on(x)  = 130.0 + 4.7 × 20.0 × x          # W, 130 W a 224 W
P_off    = 75.0                            # W, sleep mode (RF desligado)
P_cell   = P_on(x) se ON, P_off se OFF
```

Parâmetros da Tabela 1 de Auer et al. (EARTH), modelo macro:
`N = 6`, `P_max = 20 W`, `P_0 = 130 W`, `Δp = 4.7`, `P_sleep = 75 W`.
Usamos `N = 1` porque o KPI é **por sector/célula** e o cenário tem uma antena
lógica por gNB; `N = 6` fica como análise de sensibilidade.

**Fonte:** `computers-15-00050 (1).pdf` (revisão MDPI, Tabela 1) e o modelo
EARTH original (Auer et al., 2011). O artigo do CCNC não publica parâmetros; a
Fig. 3b reporta energia em **%** (Always-On = 100%), não em watts.

**Consequências esperadas:**
- a energia deixa de ser proporcional ao throughput; `corr(energia, thr)` deve
  cair de 0,893 para um valor bem menor;
- desligar uma célula poupa no máximo `(P_on - P_off)/P_on` (66,5% a carga zero,
  33,5% a carga máxima), não 100% do tráfego;
- o reward e o estado mudam de escala; **datasets antigos não são reutilizáveis**
  sem reconversão.

**Arquivos:** `src/environments/es_env.py` (`cell_kpis_from_dataframe`,
`_get_obs`, `_build_raw_kpms`), `examples/fit_reward_calibration.py`,
`examples/collect_real_dataset.py`, `src/nsoran/drl/real_data.py`.

**Ablation:** `--energy-model proxy` reproduz o comportamento anterior.

---

## D2 — Semântica de δ

**Escolha:** (a) custo denso enquanto a célula está ON, como já implementado,
com (b) impulso OFF→ON como ablation.

**Definições:**
- **denso (principal):** `δ_i(t) = 0.9^(0.01 · T_on_ms)`, em todo passo com
  `state_i = ON`, e `0` quando OFF. `T_on_ms` é o tempo desde a última transição
  para ON.
- **impulso (ablation):** `δ_i(t) = 0.9^(0.01 · T_off_ms)` **somente** no passo
  em que `state_i` passa de OFF para ON, e `0` nos demais. `T_off_ms` é o tempo
  que a célula passou desligada; quanto maior o gap, menor o custo — como na
  frase do artigo.

**Justificativa:** a fórmula do artigo usa `T_D` = "time interval in which the
cell remains active", o que favorece o custo denso; a frase sobre "small time
gap" favorece o impulso. O custo denso é o default porque é o que a fórmula
literalmente escreve e o que o código já implementa; o impulso é a leitura
alternativa, testada como ablation.

**Arquivos:** `es_env.py::_update_es_on_cost`, `es_env.py::_reset_es_tracking`,
`real_data.py`.

**Ablation:** `--es-cost-mode impulse`.

---

## D3 — População da normalização

**Escolha:** (b) normalizar **por célula** e agregar pela média
(`sum / N`, equivalente a "soma e divide por N"), preservando o intervalo
`[0, 1]` de cada componente. A normalização global fica como ablation.

**Implementação:**
- cada componente `throughput`, `energy`, `rlf`, `es_on_cost` é normalizado com
  a distribuição empírica da **célula** (chaves `throughput_cell_2`, ...,
  `energy_cell_8` no arquivo de calibração);
- `bs_on` continua global (é uma contagem de rede);
- a agregação é a média dos valores normalizados das 7 células;
- sem calibração por célula, usa-se a referência agrupada (pooled) e a escala
  `[0, 1]` é mantida dividindo as faixas de fallback por `N = 7`.

**Justificativa:** a Eq. (2) soma componentes por célula; normalizar o total
esconde a não linearidade da transformação de quantis e faz uma célula grande
dominar o reward. A média (em vez da soma) mantém a escala dos pesos da Tabela I.

**Arquivos:** `es_env.py::compute_reward`, `RewardNormalizer`,
`fit_reward_calibration.py`, `real_data.py`.

**Ablation:** `--per-cell-normalization off` (normalização global).

---

## D4 — Algoritmo offline do PPO

**Escolha:** (b) IQL (Implicit Q-Learning) para os agentes PPO-1..PPO-5.
O `OfflineReplayEnv` (replay) permanece como ablation e para compatibilidade.

**Por quê:** `OfflineReplayEnv.step()` ignora a ação do agente; o PPO treinado
nele é, na prática, behavior cloning ponderado pelo reward, não offline RL.
IQL é um algoritmo offline de ações discretas, estável, sem simulação e sem
necessidade de avaliar ações fora do dataset (usa expectil de `min(Q1,Q2)` e
extração por AWR). Não é "PPO offline" — é um agente offline distinto,
registrado como tal. O DQN continua com REM-DQN + CQL.

**Hiperparâmetros iniciais:** `gamma = 0.99`, `tau = 0.7` (expectil),
`beta = 1.0` (AWR), `lr = 3e-4`, `batch = 256`, 2 Q-heads + V + política MLP
`[256, 256]`, target update suave `0.005`, 200k gradientes, early stop por
perda de validação (paciência 4). Ajustes serão registrados em
`experiment-configs/training_matrix.json`.

**Arquivos:** novo `src/nsoran/drl/iql.py`; dispatch em `train.py`,
`evaluate.py`, `examples/_agent_cli.py`; testes em `tests/test_drl_shapes.py`.

**Ablation:** `--offline-algo replay` (comportamento anterior).

---

## D5 — Escala da campanha

**Escolha:** Tier M (554 runs de simulação, ~720 h de CPU, ~5 dias com 6
workers, ~130 GB brutos, ~3 GB retidos), executado em duas etapas:

1. **Tier S (gate):** 181 runs no total; o gate inicial é o braço B
   (baselines, 40 runs, 5 seeds) + um dataset piloto e 6 treinos.
2. **Tier M:** somente se o gate S passar nos critérios de aceitação
   (`reproduction-plan.md` §6): runs completos ≥ 95%, KPMs não degenerados,
   calibração estável, reward não colapsado.

**Orçamento:** 6 workers (≈ 11 GB de RAM), 1,3 h/run, 130 GB brutos com
retenção seletiva (5 MB/run após ingestão). O disco livre atual (207 GB) é
suficiente para o Tier M; o Tier L (28 dias) continua fora do horizonte.

**Arquivos:** `scripts/run_paper_baselines.py` (baselines),
`scripts/run_dataset_campaign.py` (novo, braço D), `scripts/run_eval_campaign.py`
(novo, braço E), manifestos em `experiment-configs/`.

---

## Escopo de código autorizado

| Repositório | Autorizado | Observação |
|---|---|---|
| `ns-o-ran-gym` | sim | ambiente, DRL, exemplos, scripts, testes |
| `ns-3-mmwave-oran` | somente se necessário | nenhum KPM novo é necessário para D1–D3; não alterar sem registrar |
| `oran-e2sim` | não | fora do caminho experimental |

## Invariantes que não mudam

- `Discrete(128)`, células `[2..8]`, observação `Box(0,1,(85,))`.
- Eq. (2) e os pesos da Tabela I.
- Sem dados sintéticos em resultados; `--synthetic` é smoke test.
- Runs abortados são reportados, não descartados em silêncio.
- Nenhum resultado é apresentado como replicação exata do artigo: é um **novo
  experimento** com especificação declarada (ver `implementation-audit.md` §4).
