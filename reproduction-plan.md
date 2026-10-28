# Plano de reprodução e experimentação

**Objetivo.** Transformar a auditoria (`implementation-audit.md`) em um plano
executável, com custo, critérios de aceitação e portões de decisão explícitos.
O plano distingue três coisas que não podem ser confundidas:

| Categoria | Definição | O que pode ser afirmado |
|---|---|---|
| **Reprodução exata** | mesmos números do artigo, mesmo código/dataset | inviável (não há artefato dos autores) |
| **Reprodução aproximada** | mesmo MDP, mesmos pesos, mesma escala; desvios declarados | “compatível com o artigo sob as hipóteses X, Y, Z” |
| **Novo experimento** | pergunta própria sobre a infraestrutura local | “nossos resultados mostram…” (sem alegar replicação) |

**Regra.** Nenhum resultado deste plano é “reprodução do artigo” até que os
critérios da §8 sejam cumpridos. Até lá, o rótulo é **novo experimento**.

---

## 0. Estado de execução (24 Out 2026)

| Item | Estado |
|---|---|
| D1 (energia EARTH em W) | ✅ implementado; default `--energy-model watts`, ablação `proxy` |
| D2 (δ denso vs. impulso) | ✅ implementado; default `dense`, ablação `impulse` |
| D3 (normalização por célula) | ✅ implementado; default `per-cell`/`mean`, ablação global |
| D4 (IQL offline para PPO-1..5) | ✅ implementado; DQN mantém REM-DQN+CQL |
| D5 (escala) | 🟡 **Tier S em curso** — apenas o braço B (40 baselines), 21 completos + 19 em falta (5 a correr, 14 em fila; 3 são reposições); braços D/T/E/A ainda não começaram |
| Dataset D1–D3 | ✅ 2.079 transições / 21 episódios (`baselines_dataset_d1d3.npz`) |
| Calibração D1–D3 | ✅ 5 séries globais + 4×7 por célula (`reward_calibration_d1d3.npz`) |
| Treino D1–D3 | 🟡 smoke IQL real de 256 passos concluído (2.079 transições, `/tmp/opencode/iql_real_smoke`); nenhum treino sério com ≥3 seeds; checkpoints antigos usam 843 transições |
| `ns-3-mmwave-oran` | ✅ não alterado para D1–D5 |
| `oran-e2sim` | ✅ não tocado |

**Leitura correcta deste estado:** o pipeline D1–D4 está implementado e validado
por um run real curto (9 indicações, 353,5 s de parede), mas a campanha de
baselines ainda está a meio. Os números de `analysis/` são o estágio de 21 runs e
serão re-agregados quando os 40 estiverem completos. Nenhum resultado de política
DRL foi produzido com D1–D4; não há base para comparar com a Tabela VI.

O estado da campanha está em
`ns-o-ran-gym/examples/output/real_runs/manifest.json`; o log é
`/tmp/opencode/tier_s_campaign.log`. O manifesto pré-campanha está em
`.../manifest_pre_tier_s.json`.

---

## 1. Ativos já existentes

| Ativo | Caminho | Volume | Estado |
|---|---|---|---|
| Runs reais de baseline | `ns-o-ran-gym/examples/output/real_runs/` | 40 jobs planeados (21 ok, 19 em falta — 5 a correr, 14 em fila —, 3 crashes históricos a repor) | utilizáveis com ressalvas; campanha Tier S em curso |
| Dataset D1–D3 | `.../real_runs/baselines_dataset_d1d3.npz` | 2.079 transições, 21 episódios | ✅ calibrado, energia watts, normalização por célula |
| Dataset antigo (ablação) | `.../real_runs/baselines_dataset.npz` | 2.079 transições, 21 episódios | energia proxy, normalização global, sem `environment_model` |
| Calibração D1–D3 | `.../reward_calibration_d1d3.npz` | 33 séries (5 globais + 4×7 por célula) | ✅ ajustada nos 21 runs reais |
| Métricas por run | `reproduction-results.csv` | 24 linhas | gerado a partir dos SQLite; energia proxy (pré-D1) |
| Cenário do artigo | `src/environments/scenario_configurations/paper_es_use_case.json` | sha256 `3def717c…83f2e8` | congelado |
| Checkpoints de smoke | `examples/paper_runs/`; `/tmp/opencode/iql_real_smoke` | PPO-2 50k, DQN 20k; smoke IQL real de 256 passos | **apenas pipeline**; 843 transições antigas + 2.079 reais; não avaliar |
| ns-3 | `ns-3-mmwave-oran/build/scratch/ns3.38.rc1-scenario-three-default` | build `default` | funcional |

**Nada disso é resultado de política.** Os 21 runs completos são baselines
scriptados; os checkpoints treinaram com 843 transições e pararam cedo. O dataset
D1–D3 serve para treinar os agentes, mas ainda não foi usado num treino completo.

---

## 2. Decisões (aprovadas e implementadas)

Estas cinco decisões mudam o significado dos resultados. Foram aprovadas em
2026-10-22 e estão registradas em `experiment-configs/decisions.md`.

| ID | Decisão | Opções | Escolha | Impacto |
|---|---|---|---|---|
| **D1** | Modelo de energia | (a) manter proxy `TB×1.0`; (b) EARTH em W; (c) E3F | **(b)** EARTH `P_on = 130 + 94·x` W, `P_sleep = 75 W`; (a) como ablação | comparação com Fig. 3b |
| **D2** | Semântica de $\delta$ | (a) custo denso enquanto ON; (b) impulso na transição OFF→ON | **(a)** denso como default; (b) como ablação | escala do termo de custo |
| **D3** | População da normalização | (a) totais da rede; (b) por célula e depois média | **(b)** por célula, agregação pela média; (a) como ablação | não linearidade |
| **D4** | Algoritmo offline do PPO | (a) manter replay; (b) CQL/IQL/BCQ/TD3+BC | **(b)** IQL discreto para PPO-1..PPO-5; DQN mantém REM-DQN+CQL | validade do agente |
| **D5** | Escala da campanha | Tier S / M / L (§5) | **Tier S** para o piloto; Tier M condicionado ao fim do Tier S; Tier L inviável | tempo e disco |

A aprovação de D1–D5 e do escopo de código foi dada; não há portão pendente. O
trabalho restante é de campanha (braço B), treino e avaliação.

---

## 3. Desenho experimental

### 3.1 Fatores

| Fator | Níveis | Como é controlado |
|---|---|---|
| Algoritmo | PPO-1..5, DQN | `--agent` / script por agente |
| Seed de treino | 3 (mínimo) ou 5 | `--seed` |
| Calibração de recompensa | fallback vs. calibrada | `--calibration-file` |
| População de normalização | total vs. por célula | código (D3) |
| Semântica de δ | densa vs. impulso | código (D2) |
| Modelo de energia | proxy vs. W | código (D1) |
| Seed de avaliação | 5 | `reset(seed=...)` → `RngRun` |

### 3.2 Braços

| Braço | Descrição | Runs de simulação | Treinos |
|---|---|---|---|
| **B** | baselines: Always-On, Random, Static×3, Dynamic×3, 5 seeds | 40 | — |
| **D** | dataset de comportamento (keep/random/random-seconds/static/dynamic) | 400 | — |
| **T** | 6 agentes × 3 seeds | — | 18 |
| **E** | avaliação live: 18 políticas × 5 seeds | 90 | — |
| **A1** | calibração on/off (PPO-2, 3 seeds) | 6 | 6 |
| **A2** | normalização total vs. por célula (PPO-2, 3 seeds) | 6 | 6 |
| **A3** | δ denso vs. impulso (PPO-2, 3 seeds) | 6 | 6 |
| **A4** | energia proxy vs. W (PPO-2, 3 seeds) | 6 | 6 |

**Total Tier M:** 40 + 400 + 90 + 24 = **554 runs de simulação** e
**42 treinos offline**.

> Os 40 baselines incluem os 21 já existentes; os 3 runs que falharam precisam ser
> refeitos. Runs novos de baseline: **19**.

### 3.3 Por que 400 runs de dataset

O artigo reporta “~3.000 simulações / >300.000 data points”. Note que

$$
3.000\ \text{runs}\times 100\ \text{indicações}=300.000,
$$

o que sugere que “data points” são **amostras por indicação**, não simulações
independentes. Ainda assim, 3.000 runs são inviáveis aqui (§5). Com 400 runs:

$$
400\times 99 = 39.600\ \text{transições}\approx 13\%\ \text{da escala do artigo}.
$$

Isso é suficiente para treinar e comparar algoritmos, **não** para alegar
equivalência de escala. A limitação deve entrar em qualquer texto derivado.

---

## 4. Protocolo por fase

### Fase 0 — sanidade (1 h, 1 run) — ✅ concluída

O smoke real de 1 s correu 9 indicações em 353,5 s de parede, sem truncamento,
com `database.db` a conter `gnb_cu_cp`/`gnb_cu_up`/`du` para as células 2–8 e
`bsState.txt` com as 7 células. A conversão deu 9 transições `(9, 85)`, acções
`[127, 127]` e metadados `watts`/`dense`/`per-cell`. Isto valida o pipeline; não
é resultado de política.

```bash
cd /mnt/legitbet/elioth_sbrc/ns-o-ran-gym

# 1. suíte rápida
.venv/bin/python -m pytest -q \
  tests/test_es_env_unit.py tests/test_real_data.py \
  tests/test_drl_shapes.py tests/test_agent_scripts.py

# 2. run curto real (NÃO usar --synthetic para resultados)
.venv/bin/python examples/collect_run.py \
  --config src/environments/scenario_configurations/paper_es_use_case.json \
  --ns3_path ../ns-3-mmwave-oran \
  --output_folder examples/output/sanity \
  --policy keep --seed 900 --num-steps 2
```

**Critério de saída:** 2 indicações, `database.db` com tabelas `gnb_*`/`du`
preenchidas para as células 2–8, `bsState.txt` com 7 células.

### Fase 1 — decisões e calibração (2–3 dias, 0 runs novos) — ✅ concluída

1. D1–D5 registrados em `experiment-configs/decisions.md`.
2. Calibração D1–D3 nos 21 runs completos:

```bash
.venv/bin/python examples/fit_reward_calibration.py \
  --runs-dir examples/output/real_runs \
  --config src/environments/scenario_configurations/paper_es_use_case.json \
  --energy-model watts --es-cost-mode dense --per-cell-normalization \
  --output examples/output/reward_calibration_d1d3.npz
```

> `--runs-dir` inclui **todas** as subpastas, inclusive runs abortados. Depois da
> campanha, refazer com `--run-dir` explícito para cada run com `status=ok` no
> manifesto; caso contrário a calibração aprende com transições truncadas.

3. Reconverter o dataset com a calibração e o modelo D1–D3:

```bash
.venv/bin/python examples/collect_real_dataset.py \
  --runs-dir examples/output/real_runs \
  --config src/environments/scenario_configurations/paper_es_use_case.json \
  --agent ppo-2 \
  --energy-model watts --es-cost-mode dense --per-cell-normalization \
  --calibration-file examples/output/reward_calibration_d1d3.npz \
  --output examples/output/real_runs/baselines_dataset_d1d3.npz
```

**Resultado:** 2.079 transições / 21 episódios; 33 séries de calibração (5 globais
+ 4×7 por célula); Eq. (2) reconstruída com erro `4.77e-08`. `bs_on` caiu de
~89,7% para **40%** da massa negativa (limiar de invalidade: >80%). A correlação
`corr(reward, bs_on)` continua **−0,953**, acima do alvo de 0,8: a normalização
removeu a dominância da *média*, mas `bs_on` ainda tem a maior *dispersão*
(dp 0,278). O critério de saída foi cumprido no primeiro ponto e não no segundo;
isto fica registrado como limitação, não como sucesso.

### Fase 2 — baselines em 5 seeds (2 dias, 19 runs) — 🟡 21/40 em curso

```bash
.venv/bin/python scripts/run_paper_baselines.py \
  --config src/environments/scenario_configurations/paper_es_use_case.json \
  --ns3-path ../ns-3-mmwave-oran \
  --output-dir examples/output/real_runs \
  --seeds 555 556 557 558 559 \
  --jobs 5 --skip-build --min-free-mb 3500
```

O comando salta os 21 jobs já completos e corre os 19 em falta (seeds 558–559 e
reposição de `dynamic_2_2_0_3_seed555/556` e `static_2_2_0_3_seed556`). O
`--jobs 5` é o limite de RAM (1,8 GB/run × 5 ≈ 9 GB de 15 GiB); `--skip-build`
reutiliza o build `default`; `--min-free-mb 3500` impede arrancar runs que seriam
mortos por OOM.

**Critério de saída:** 40/40 runs completos, `status=ok` no manifesto, nenhum
crash, `analysis/` re-agregado com 40 linhas e janela de 99 indicações. **Ainda
não cumprido:** 21/40.

### Fase 3 — coleta do dataset (5 dias, 400 runs) — ❌ não iniciada

```bash
.venv/bin/python scripts/collect_dataset_campaign.py \
  --config src/environments/scenario_configurations/paper_es_use_case.json \
  --ns3-path ../ns-3-mmwave-oran \
  --output-dir examples/output/dataset_runs \
  --plan experiment-configs/dataset_plan.json \
  --workers 5
```

> O script `collect_dataset_campaign.py` **não existe** hoje; precisa ser criado
> (ou a campanha precisa ser dirigida por um loop em `collect_run.py`). Isso é
> parte do trabalho autorizado na Fase 3, não um pressuposto. O valor `--workers 5`
> é o mesmo limite de RAM da campanha actual.

**Critério de saída:** ≥380 runs ok (95%), cobertura das 5 behavior policies,
`manifest.json` com seed e comando de cada run.

### Fase 4 — treino offline (1–2 dias, 42 treinos) — ❌ não iniciada

O dataset D1–D3 disponível tem **2.079 transições** (21 runs), muito abaixo das
39.600 transições que 400 runs produziriam. Os comandos abaixo são o exemplo do
protocolo; ainda não foram executados com D1–D3.

```bash
# PPO (exemplo: PPO-2, seed 0) — dataset D1–D3 actual (2.079 transições)
.venv/bin/python examples/ppo_2.py \
  --dataset examples/output/real_runs/baselines_dataset_d1d3.npz \
  --output-dir examples/output/training/PPO-2_s0 \
  --total-timesteps 1200000 --seed 0 --device cpu

# DQN
.venv/bin/python examples/dqn.py \
  --dataset examples/output/real_runs/baselines_dataset_d1d3.npz \
  --output-dir examples/output/training/DQN_s0 \
  --total-timesteps 200000 --seed 0 --device cpu
```

**Critério de saída:** 42/42 treinos concluídos ou parados pelo early stop;
`metadata.json` com timesteps, melhor recompensa e tempo.

### Fase 5 — avaliação live (5 dias, 90 + 24 runs) — ❌ não iniciada

```bash
.venv/bin/python examples/ppo_2.py \
  --eval-only \
  --model examples/output/training/PPO-2_s0/best_model.zip \
  --ns3-path ../ns-3-mmwave-oran \
  --scenario-config src/environments/scenario_configurations/paper_es_use_case.json \
  --ns3-output-folder examples/output/eval/PPO-2_s0_r555 \
  --episodes 1 --seed 555
```

**Atenção.** `--env dataset` (sem `--ns3-path`) apenas reproduz transições
gravadas; **não é avaliação de política**. Toda avaliação de política usa
`--ns3-path`.

**Critério de saída:** 90/90 avaliações completas; throughput/energia/RLF por
indicação registrados; nenhuma execução usa dataset replay.

### Fase 6 — análise (2–3 dias) — 🟡 parcial

1. Agregar com `examples/compare_runs.py --window 99`.
2. Gerar CDFs (não só médias) de throughput, energia, RLF e `bs_on`.
3. Comparar com as Figs. 3–5 do artigo **qualitativamente** (ordem, forma), não
   numericamente.
4. Rodar os contrastes A1–A4 e reportar tamanho de efeito.
5. Escrever `analysis/reproduction-report.md` com a tabela de desvios.

---

## 5. Tiers de escala

Custo por run: **~78 min** de parede (faixa medida 68–85 min). A campanha em
execução usa **5 workers** (1,8 GB/run × 5 ≈ 9 GB de 15 GiB); 6 workers é o
máximo teórico, não o valor usado.

| Tier | Runs de simulação | Treinos | Horas de CPU | Dias (6 workers) | Disco bruto | Disco retido | Verdict |
|---|---|---|---|---|---|---|---|
| **S** — mínimo | 181 | 12 | ~235 h | ~1,7 | ~45 GB | ~1 GB | publicável como “estudo piloto” |
| **M** — recomendado | 554 | 42 | ~720 h | ~5,0 | ~130 GB | ~3 GB | adequado a dissertação |
| **L** — escala do artigo | 3.130 | 42 | ~4.070 h | ~28 | ~730 GB | ~16 GB | **inviável por tempo** (~1 mês) |

**Execução actual.** O que está a correr é apenas o **braço B do Tier S** (40
baselines em 5 seeds), não os 181 runs do Tier S completo: os braços D (400 runs
de dataset), T (18 treinos), E (90 avaliações) e A1–A4 (24 runs/treinos) ainda não
começaram. Em 24 Out 2026: 21/40 baselines completos, 19 em falta (5 a correr, 14
em fila).

**Cálculo.** `horas = runs × 1,3 h`; `dias = horas / 6 / 24`. O “disco bruto” conta
os traces do ns-3 (~233 MB/run); o “disco retido” conta só `database.db` (2,4 MB) +
`bsState.txt` + `EsActions.txt` + log + manifesto (~5 MB/run), após ingestão
validada. Com retenção selectiva o disco **não** bloqueia o Tier L; o tempo sim.

**Recomendação.** Terminar o braço B do Tier S e usá-lo como piloto; só depois
decidir se o Tier M cabe no calendário. O Tier L continua fora do horizonte de
mestrado por exigir ~1 mês de CPU dedicado, mas já não por falta de disco.

---

## 6. Métricas

### 6.1 Por run (primárias)

| Métrica | Definição | Fonte |
|---|---|---|
| throughput | média de `QosFlow.PdcpPduVolumeDL_Filter` por indicação | `grafana` / `compare_runs.run_metrics` |
| energia | EARTH em W: `P_on = 130 + 94·x`, `P_sleep = 75 W` (D1); `TB.TotNbrDl.1 × p_tx` como ablação | idem |
| RLF | média e máximo de UEs com SINR < −5 dB | idem |
| `bs_on` | média de células ON | `bsState.txt` |
| custo de ativação | média de $\delta$ | `grafana.on_cost` |
| razão energia/throughput | derivada | análise |

### 6.2 Por política (secundárias)

- CDF de throughput, energia, RLF e `bs_on` (não apenas média).
- Delta percentual vs. Always-On, com IC bootstrap (10.000 reamostragens).
- Tamanho de efeito (Cliff's delta) entre políticas.
- Janela fixa de 99 indicações para comparabilidade.

### 6.3 De treino

- Curva de recompensa por avaliação.
- Melhor recompensa e timestep.
- Estabilidade entre seeds (desvio-padrão da recompensa final).
- Para DQN: TD loss e CQL loss.

---

## 7. Riscos

| Risco | Prob. | Impacto | Mitigação |
|---|---|---|---|
| Crash residual LTE/TCP (3/24 históricos) | média | perda de runs | correções SACK/PDU já no fork; campanha repõe os 3 com manifesto e orça 15% de retrabalho |
| Disco insuficiente | média | campanha para | retenção seletiva (manter `database.db`, `bsState.txt`, `EsActions.txt`) |
| RAM no limite | média | runs mortos por OOM | 5 workers na campanha actual; gate `--min-free-mb 3500`; monitorar RSS |
| IQL offline mal ajustado | média | política fraca | D4 implementado; validar curvas e comparar com replay como ablação |
| Dataset D1–D3 pequeno (2.079 transições) | **alta** | overfitting / baixa cobertura | não comparar com os 300k do artigo; usar como piloto e expandir para 400 runs se o tempo permitir |
| Calibração mal ajustada | baixa | recompensa degenerada | calibração D1–D3 feita; validar decomposição antes de treinar |
| `bs_on` ainda domina a variância | **alta** | reward não reflecte QoS | documentado; reportar reward **e** throughput/RLF/energia; não usar só o reward como critério |
| Semáforos órfãos | baixa | run trava no início | `close()` idempotente; limpar `/dev/shm/sem.*` |
| Tempo de mestrado | alta | plano incompleto | braço B do Tier S como entrega mínima; análise parcial explicitamente rotulada |

---

## 8. Critérios de aceitação

### 8.1 Para dizer “reprodução aproximada”

Todos os itens:

| # | Critério | Estado em 24 Out 2026 |
|---|---|---|
| 1 | D1–D5 registrados e justificados | ✅ `experiment-configs/decisions.md` |
| 2 | Dataset calibrado; `bs_on` sem dominância (>80% é falha) | ✅ 40% da massa negativa; ⚠️ `corr(reward, bs_on) = −0.953` e a dispersão de `bs_on` ainda domina |
| 3 | 40 baselines completos em 5 seeds, sem crash | 🟡 21/40; campanha em curso |
| 4 | 6 agentes treinados com ≥3 seeds | ❌ nenhum treino D1–D3 sério; smoke IQL real de 256 passos passou |
| 5 | 90 avaliações live (não replay) | ❌ não iniciadas |
| 6 | CDFs comparadas com as Figs. 3–5; diferenças explicadas | ❌ não geradas |
| 7 | Tabela de desvios (R1–R9) com impacto estimado | 🟡 R1–R9 em `mathematical-formulation.md`; impacto quantitativo por desvio ainda incompleto |

**Conclusão:** os critérios 1–2 estão cumpridos com a ressalva do critério 2;
3–7 não estão. O rótulo continua **novo experimento**, não “reprodução aproximada”.

### 8.2 Para dizer “novo experimento”

Basta: hipótese própria, baseline próprio, dados reais de ns-3, seeds declaradas,
e **não** afirmar replicação.

### 8.3 Sinais de que o resultado é inválido

- recompensa dominada por um único termo (> 80%);
- throughput e energia com correlação > 0,85 (cancelamento);
- avaliação feita com `--env dataset`;
- runs com menos de 99 indicações incluídos na comparação;
- seeds ausentes ou não registradas;
- dados sintéticos em qualquer tabela de resultados.

**Checks actuais:** `bs_on` = 40% (não dispara), `corr(thr, energy) = 0.781`
(não dispara), 3 runs abortados excluídos, 5 seeds declaradas, nenhum dado
sintético. O sinal de alerta que **dispara** é qualitativo: o reward ainda está
anti-correlacionado com o throughput (`−0.527`), pelo que o reward não pode ser a
única métrica de sucesso.

---

## 9. Cronograma real (Tier S, braço B)

| Data | Atividade | Estado |
|---|---|---|
| 21–22 Out 2026 | Fases A/B/F, 21 baselines reais, análise quantitativa inicial | ✅ concluído |
| 23 Out 2026 | D1–D4 implementados, teste unitário e smoke real de 1 s | ✅ concluído |
| 24 Out 2026 | Campanha dos 19 baselines em falta; calibração/dataset D1–D3; re-análise | 🟡 21/40 completos, 19 em falta (5 a correr, 14 em fila) |
| Após a campanha | Re-ajustar calibração só com runs `status=ok`, reconverter o dataset dos 40 runs, re-agregar baselines e reward | ⏳ pendente |
| Se o calendário permitir | Fase 3 (400 runs), Fase 4 (42 treinos), Fase 5 (90 avaliações live) | ❌ não iniciado |
| — | Tier M (554 runs) e Tier L (3.130 runs) | ❌ não agendados; Tier L inviável (~28 dias) |

O caminho mínimo entregável é o braço B do Tier S + a análise D1–D3. O caminho
completo para “reprodução aproximada” exige os critérios §8.1 e não cabe no
calendário sem uma decisão explícita sobre o Tier M.

---

## 10. Entregáveis

| Arquivo | Conteúdo |
|---|---|
| `experiment-configs/decisions.md` | D1–D5 registrados |
| `experiment-configs/baselines_5seeds.json` | matriz de baselines |
| `experiment-configs/dataset_plan.json` | composição dos 400 runs |
| `experiment-configs/training_matrix.json` | 42 treinos |
| `experiment-configs/eval_matrix.json` | 90 + 24 avaliações |
| `reproduction-results.csv` | métricas por run (atualizado) |
| `analysis/baseline_summary.csv` | agregados por política |
| `analysis/reward_decomposition.csv` | decomposição da recompensa |
| `analysis/reproduction-report.md` | relatório final com desvios |
| `analysis/figures/` | CDFs e curvas de treino |

---

## 11. Checklist de aprovação e execução

**Aprovação (24 Out 2026):**

- [x] D1 (modelo de energia) escolhido: EARTH em watts
- [x] D2 (semântica de δ) escolhido: denso, com impulso como ablação
- [x] D3 (população de normalização) escolhido: por célula, média
- [x] D4 (algoritmo offline do PPO) escolhido: IQL
- [x] D5 (Tier S/M/L) escolhido: Tier S (braço B) para o piloto; Tier M condicionado
- [x] Autorização explícita para editar `ns-o-ran-gym`
- [x] `ns-3-mmwave-oran` não foi alterado para D1–D5; `oran-e2sim` não foi tocado
- [ ] `scripts/collect_dataset_campaign.py` criado — **não necessário nesta fase**; a Fase 3 não começou
- [x] Ciência de que o Tier M custa ~5 dias de CPU e ~130 GB; o Tier S foi escolhido
- [x] Ciência de que o resultado será “novo experimento”, não replicação

**Execução:**

- [x] Fase 0: smoke real de 1 s (9 indicações, 353,5 s)
- [x] Fase 1: D1–D3 implementados; calibração e dataset D1–D3 (2.079 transições)
- [ ] Fase 2: 40/40 baselines (21/40 em 24 Out 2026)
- [ ] Fase 3: 400 runs de dataset
- [ ] Fase 4: 42 treinos D1–D3
- [ ] Fase 5: 90 avaliações live
- [ ] Fase 6: CDFs, relatório final e tabela de desvios quantificada

**O plano está aprovado e em execução parcial.** O que falta é campanha, treino e
avaliação — não decisões.
