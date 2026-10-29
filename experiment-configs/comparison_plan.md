# Plano de comparação — Baseline (artigo base) vs. Nossa proposta (CMDP/Offline RL)

**Data:** 29 de outubro de 2026 · **Status:** registro vinculante de desenho experimental
**Entradas:** `decisions.md` (D1–D5) · `research-extensions.md` (E1/E2/E7) · `cmdp_design.md` (Rev. 1, A1–A5) · `training_matrix.json` / `eval_matrix.json` / `dataset_plan.json` / `cost_model.json` · custos medidos em `examples/output/training/tabela_i_s0/training_cost_s0.csv`
**Política de dados:** 100% ns-3 real; treino offline com dataset real; avaliação sempre **live**; `--synthetic` proibido em resultados.

---

## 1. Pergunta de pesquisa e hipóteses (pré-registradas)

**Pergunta central:** *Uma política treinada com restrição explícita de QoS (CMDP) sobre
offline RL genuíno domina os agentes da Eq. (2) do artigo base e as heurísticas de [3]
na fronteira energia–vazão–RLF?*

| # | Hipótese | Predição falsificável |
|---|---|---|
| H1 | Os agentes da Eq. (2) sem restrição caem na zona de colapso | ΔThr ≤ −20% para ≥ 4 dos 6 agentes, mesmo com reward calibrado (D3) |
| H2 | CMDP pousa na região viável alta | ∃ λ da grade com ΔE ≥ +10% e ΔThr ≥ −10% e RLF máx ≤ 2 (IC95, 5 seeds) |
| H3 | O algoritmo importa quando o dataset é fraco | BC-replay < IQL ≈ CQL em avaliação live com dataset Tier S; a diferença estreita no Tier M |
| H4 | Economia sem QoS é tráfego não entregue | Nos pontos dos agentes da Eq. (2): E/thr > 1 e ΔThr < −ΔE (mesma diagonal 1:1 dos baselines) |
| H5 | O custo computacional do offline é desprezível vs. simulação | treino ≤ 1 CPU-h/agente vs. ≥ 1,3 h-máquina por avaliação live (já medido: 0,63–0,71 CPU-h) |
| H6 | Dormir células degrada latência além da vazão | p95 de `DRB.PdcpSduDelayDl` dos agentes Eq. (2) e baselines agressivos ≥ 2× Always-On (sinal já medido: Random p95 946 vs. 162 Always-On, 5,8×; Static (4,2,1) 267) |

Se H2 falhar (nenhum λ cumpre), o resultado reportado é a **fronteira de inviabilidade**
(até onde dá para ir sem quebrar o SLA) — resultado negativo útil, declarado no
`research-extensions.md` §E1.

## 2. Contrato de comparação justa (apples-to-apples)

1. **Mesmo dataset** para todos os agentes treinados (Tier S → Tier M, mesmo `.npz`,
   mesma calibração quantílica D3 aplicada via `reward_components`).
2. **Mesmo modelo de energia (D1 watts)** para baselines e agentes — nenhuma métrica
   usa o proxy da Eq. (1).
3. **Mesmo orçamento de treino:** 1,2 M passos com o early stop do artigo
   (paciência 4, mín. 10k) para todos; DQN 200k. Sem tuning por braço.
4. **Duas condições para o baseline "artigo"** (declarar ambas):
   - **B1 artigo-como-publicado:** Eq. (2) com normalização fallback (sem calibração);
   - **B2 artigo-corrigido:** Eq. (2) calibrada D3 (a condição cientificamente justa;
     é a que compete com a nossa proposta). B1 entra só para quantificar o efeito
     da calibração (Fig. 4 do pitch).
5. **Mesma avaliação:** live no ns-3, cenário paper, 5 seeds (555–559), 99 indicações,
   janela integral; aggregates via `compare_runs.py`; **latência PDCP como métrica de
   QoS de primeira classe** (mediana + p95 por run, das colunas
   `DRB.PdcpSduDelayDl(cellAverageLatency)` / `...UEID(pdcpLatency)` dos
   `cu-up-cell-*.txt` — 6.237 amostras por run = 63 UEs × 99 indicações, sem
   missing; habilitadas na ingestão pela Fase 0c).
6. **Nada de replay como avaliação** (aviso do `eval_matrix.json`).

## 3. Braços experimentais

| Braço | Grupo | Configuração | Treinos (3 seeds) | Evals live (5 seeds) |
|---|---|---|---|---|
| **A — Referência** | Always-On, Static×3, Dynamic×3, Random | heurísticas ns-3 (já 34/40 → 40/40) | 0 | 0 (campanha Tier S serve) |
| **B1 — Artigo, como publicado** | PPO-2 (representante) | Eq. (2), fallback, estado 85 | 3 | 5 |
| **B2 — Artigo, corrigido** | PPO-1..5 (IQL, pesos Tabela I) + DQN (REM+CQL) | Eq. (2) calibrada D3, estado 85 | 6×3 = 18 | 6×5 = 30 |
| **C — Controle de algoritmo** | BC-replay | `--offline-algo replay`, Eq. (2) D3 | 3 | 5 |
| **D — Nossa proposta** | IQL + CMDP-Lagrangiano (grade λ ∈ {0; 0,5; 1; 2; 5; 10}) | Eq. (2) D3 + `constraint_mode=lagrangian`, `compact70+gate` (71 dims) | 6×3 = 18 (seleção offline) | top-2λ × 5 = 10 |
| **D' — Nossa proposta (via 2)** | IQL + porta suave Eq. (5.1′) | `constraint_mode=barrier`, τ_SLA da calibração | 3 | 1 config × 5 = 5 |
| **E — Ablações** | CQL+CMDP (1 λ); porta dura vs. suave; estado 85 vs. 70 | conforme `cmdp_design.md` §8 | ~9 | 2×5 = 10 |

**Seleção sem vazamento:** o λ do braço D é escolhido por validação *offline*
(reward de validação + custo estimado do SLA), nunca pelo resultado live; os top-2λ
é que vão para os 10 runs de avaliação.

## 4. Métricas e critérios de aceitação (pré-registrados)

**Primário (critério de sucesso da proposta):**
ΔE ≥ +10% vs. Always-On **com** ΔThr ≥ −10% **e** RLF máx ≤ 2 UEs, IC95 (bootstrap
10k) sobre 5 seeds de avaliação, ≥ 2 seeds de treino alcançando.

**Secundários:**
intensidade E/thr (alvo ≤ 1,05); **latência PDCP: mediana e p95 por run, critério
p95 ≤ 1,25× Always-On** (e relatório da mediana); violações de SLA por episódio
(→ 0); hipervolume da fronteira (dominância vs. Static/Dynamic); distância TV à
behavior policy; ações OOD; custo computacional (CPU-h de treino e h-máquina de
avaliação — já instrumentado).

**Comparação estatística entre braços:** bootstrap da diferença vs. o melhor baseline
viável (esperado Static 4-2-1) e vs. o melhor agente B2; reportar IC95 das diferenças;
com ruído de seed conhecido (±10–19 pp em vazão), nenhuma alegação de superioridade
com IC cobrindo zero.

## 5. Dataset, por fase

| Fase | Dataset | Transições | Uso |
|---|---|---|---|
| 1 (hoje) | `baselines_dataset_d1d3.npz` | 2.079 | pipeline validado; treinos seed 0 feitos (50k steps, early stop) |
| 2 | 40/40 Tier S reconvertido | ~4.000 | primeira comparação B2 vs. D |
| 3 | Tier M (400 runs, `collect_dataset_campaign.py` a criar) | ~39.600 | comparação final (N/d ≈ 57 com compact70) |

## 6. Execução, orçamento e comandos

| # | Passo | Máquina (6 workers) | Comando/gap |
|---|---|---|---|
| 0a | Rerun 6 crashes → 40/40 | ~9 h | `run_paper_baselines.py --only <6 jobs> --force` |
| 0b | CLI CMDP + script da grade λ + coletor Tier M | ~1 dia de código | ✅ feitos (28/10: `_agent_cli.py` grupo CMDP, `run_cmdp_lambda_grid.py`, `collect_dataset_campaign.py`) |
| 0c | **Latência nível 1**: fix do `datalake.py` (`gnb_cu_up_keys` + as 2 colunas de delay) + agregação mediana/p95 em `recompute_baselines.py`/`compare_runs.py` (lê os `.txt` dos 40 runs existentes — sem simulação nova) | ~meio dia de código, 0 h de máquina | gap mapeado; unidade (ms) a validar contra `lte_cu_up` |
| 1 | Avaliação live dos 6 agentes seed 0 (primeiro Pareto com DRL) | 6×1,3 h ≈ 8 h | `ppo_N.py --eval-only --model ... --ns3-path ... --seed 555..559` |
| 2 | Seeds 1–2 dos braços B2/C/D (~45 treinos) | ~35 CPU-h ≈ 6 h wall paralelo | runner `tabela_i` estendido |
| 3 | Coleta Tier M | ~520 h / 6 ≈ 3,6 dias | `collect_dataset_campaign.py` |
| 4 | Retreino final no Tier M + seleção λ offline | ~40 CPU-h ≈ 7 h | idem |
| 5 | Avaliação live final (~50 runs: 30 B2 + 20 D/D'/E) | ~65 h ≈ 11 h | `eval_matrix` estendida |
| 6 | Análise: Tabela III, Fig. 3 final, tabelas de custo | ~1 dia humano | `compare_runs.py` + `pareto_build.py` |

**Total: ~7–9 dias de máquina**, dos quais ~85% é simulação (eval + coleta); todo o
treino somado fica abaixo de ~2 dias de CPU. Custos de referência já medidos:
treino 0,63–0,71 CPU-h/agente, 322–361 MB, CPU-only, early stop em 50k;
avaliação live ~90 min/run.

## 7. Entregáveis do artigo

1. **Tabela III (matriz):** A + B1/B2 + C + D/D' com ΔE, ΔThr, RLF, E/thr,
   **p95 de latência PDCP**, violações SLA, custo — colunas B1 vs. B2 quantificam o
   efeito da calibração; a coluna de latência quantifica H6.
2. **Fig. 3 definitiva:** fronteira com baselines (reais), agentes B2 (esperados na
   zona de colapso — confirmação de H1/H4) e os pontos CMDP (H2); **painel (b) da
   mesma grade com ΔE × p95 de latência** (a Random em 946 vs. 162 do Always-On é o
   marcador da zona de colapso em latência).
3. **Figura de custo computacional:** treino (CPU-h) vs. avaliação (h-máquina) por
   braço — o argumento de eficiência do offline.
4. **Frase-alvo do abstract:** "enquanto os agentes da Eq. (2) reproduzem o colapso
   de tráfego das heurísticas, o agente restrito entrega ΔE = X% com ΔThr = Y%
   (IC95), na região vazia da fronteira."

## 8. Riscos e salvaguardas

| Risco | Salvaguarda |
|---|---|
| Nenhum λ cumpre o SLA (H2 falha) | reportar fronteira de inviabilidade; subir τ_SLA gradualmente e documentar |
| Grade λ saturada (todos em λ=10 sem violação) | refinamento λ ∈ {7,5; 8,5; 9; 9,5} (A2 adaptativo só se necessário) |
| SIGSEGVs SACK recorrerem na campanha | protocolo gdb batch; runs reportados, nunca ocultos |
| Vazão com ±20 pp de ruído | 5 seeds + IC95; nenhuma conclusão single-seed (regra do `RELATORIO` §3.3) |
| Dataset Tier M não cobrir ações de colapso para o CMDP punir | mistura `random-seconds` (já no plano) + relatório de cobertura de ações |
| Comparação injusta com o artigo | contrato §2: B2 corrigido é o baseline que compete; B1 só quantifica o viés |
| Latência: unidade/tradução errada | validar ms contra `lte_cu_up` (5 min) antes de publicar; cauda pesada → reportar mediana + p95, nunca só média |
| Latência no treino (níveis 2/3) | **decisão revista com dados (29/10):** nos 34 runs, corr(p95\_lat, thr) $=-0{,}52$ e $R^2 = 0{,}27$ — a restrição de vazão **não** captura a cauda de latência (caso: `dynamic_4_2_0_1_seed559`, p95 = 936 ms com ~83% da vazão Always-On; chattering). Plano: **nível 2 no 2º ciclo** (estado `compact70+lat` = 77 dims, re-conversão offline, sem runs novos); **nível 3 via porta suave composta** (vazão E latência no mesmo gate da Eq. 5.1′ — sem virar grade $\lambda\times\lambda_{\text{lat}}$); $\lambda_{\text{lat}}$ independente só se o Tier M indicar |
