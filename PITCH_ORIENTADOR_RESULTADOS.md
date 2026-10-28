# PITCH_ORIENTADOR_RESULTADOS

**Data:** 28 de outubro de 2026 · **Projeto:** Energy Saving em O-RAN com Offline RL (ns-3 real)
**Papel:** documento executivo para o orientador — diagnóstico + solução em um único artigo (alvo: IEEE Transactions / conferência flagship).
**Política de dados:** todos os números abaixo vêm exclusivamente dos artefatos reais do repositório — ns-3 via Gym, sem `--synthetic`. Nada foi simulado fora do ns-3 ou copiado de tabela agregada sem rastreio. Onde o treino final (E1/E2) ainda não rodou, há um placeholder `[TARGET EXP]` com o comportamento esperado.

**Fontes primárias (rastreáveis):**

| Fonte | Conteúdo |
|---|---|
| `analysis/baseline_summary.md` / `.csv` | 21 runs reais (D1 watts): throughput/energia/RLF/Δ%/E-thr com IC95 |
| `analysis/reward_decomposition.md` / `.csv` | Decomposição da Eq. (2) em 2.079 transições reais, pré e pós-calibração D1–D3 |
| `analysis/reward_weights_sensitivity.csv` | Ranking das 4 políticas sob os 6 pesos da Tabela I do artigo |
| `analysis/baseline_summary_proxy.csv` | Mesmas 21 runs com o proxy `TB×1.0` (ablação D1) |
| `ns-o-ran-gym/examples/output/analysis/pareto_points.csv` | Campanha Tier S 34/40 (5 seeds/política, modelo D1) + pontos publicados do artigo |
| `implementation-audit.md` | Auditoria componente a componente (27 ✅ / 12 ◐ / 4 ✗ / 6 ⚠) e decisões D1–D5 |
| `experiment-configs/decisions.md` | Registro vinculante D1–D5 (modelo EARTH, δ, normalização, IQL, escala) |
| `ns-o-ran-gym/docs/plano_tecnico/` | Triagem dos 6 crashes (3 `NS_ASSERT` + 3 `SIGSEGV`), patch IHL aplicado, design CMDP, auditoria KPM 85→70 |

---

## 1. O "Elevator Pitch" para o Orientador

**O Problema Descoberto.** A literatura de energy saving em O-RAN — incluindo Bordin et al. (CCNC 2025), nosso artigo base — mede sucesso como "redução de consumo" sem separar duas coisas radicalmente diferentes: operar a rede de forma mais eficiente e simplesmente **parar de entregar tráfego**. No artigo base, a própria função de recompensa (Eq. 2) pontua o agente por reduzir o número de células ativas (`bs_on`) e um proxy de "energia" proporcional aos transport blocks carried — ou seja, o caminho de maior recompensa é desligar células e carregar menos dados. Nosso re-auditório com 2.079 transições reais de ns-3 mostra que mesmo *depois* de corrigir a escala da recompensa (calibração quantílica D3 + energia em watts EARTH D1), a variável `bs_on` — uma variável de **estado**, não de desempenho — responde sozinha por 40% de toda a massa negativa do reward (era 89,7% antes da calibração) e por `corr(reward, bs_on) = −0,953`. O resultado é matemático, não empírico: a política **Random** — a pior em throughput (−59%) e RLF (16,3 médio, máx. 44) — é o ótimo da Eq. (2) em **5 dos 6 conjuntos de pesos da Tabela I** do artigo.

**A Prova Empírica (o "Smoking Gun").** Recompusemos os baselines a partir de 21 (agora 34) runs reais de ns-3, 10 s de cenário paper, 99 indicações E2 por run, sem nenhum número copiado de tabela pronta. Com o modelo de energia em watts (EARTH macro: `P_on = 130 + 94·x` W, `P_sleep = 75` W), toda política que desliga células "economiza" **menos energia do que perde throughput**, e a intensidade energética E/thr **piora em todas elas**: Always-On = 1,000; Static = 1,093 (+9,3% de watts por bit entregue); Dynamic = 1,148; Random = 1,979 — quase o dobro de energia por bit. A Random reduz 18,96% de watts entregando 59,04% menos tráfego. A poupança do artigo base é, na prática, tráfego não entregue. Bônus de rigor: dos 40 runs da campanha Tier S, os **6 crashes concentram-se 100% nas políticas com células em sleep** (3 asserts de buffer + 3 SIGSEGV), com causa-raiz datada e patch já aplicado — desligar células sem restrição não só derruba throughput, desestabiliza a própria pilha de protocolos.

**A Proposta Vencedora (o Artigo Completo).** Não estamos fazendo uma reprodução: estamos propondo o **benchmark QoS-aware que o artigo base declara na avaliação mas não usa no treino**. O plano é um artigo de diagnóstico + solução: (i) **Diagnóstico** — decomposição analítica do reward hacking na Eq. (2), métrica de intensidade energética E/thr, ablação das decisões metodológicas ocultas (proxy vs. watts, normalização por totais vs. por célula) que invertem rankings; (ii) **Solução** — offline RL genuíno (o "PPO offline" do artigo era behavior cloning ponderado: o replay ignora a ação do agente; nós implementamos IQL discreto, decisão D4) com **restrição explícita de QoS via CMDP** (Lagrangiano primal-dual e barreira/limiar, `constraint_mode` já implementado), estado compactado 85→70 por SVD (posto completo), avaliação sempre live em ns-3 com 5 seeds e IC95. A Fig. 3 do nosso artigo — a fronteira de Pareto com piso de SLA — é a figura que o artigo base não pôde fazer, porque sua função objetivo não codifica QoS.

---

## 2. Plano de Figuras Principais (o "Storytelling" Visual do Artigo)

> Convenção deste plano: ΔE% positivo = **redução** de energia vs. Always-On (como no Pareto); Δthr% negativo = perda de throughput.

### Figura 1 — A Ilusão da Economia de Energia (ΔE vs. Δthr e Intensidade E/thr)

- **Objetivo:** provar que os baselines do artigo base reduzem watts *porque param de entregar tráfego* — e que a eficiência por bit piora sempre.
- **Formato:** barras agrupadas por política (Always-On, Static, Dynamic, Random; painel b com as 6 tuplas) + marcador lateral da razão E/thr.
  - **Eixo X:** política (categórico; painel b: tuplas ordenadas de 4-2-0-1 a 2-2-0-3).
  - **Eixo Y (primário):** ΔEnergia (%) e ΔThroughput (%) — pares de barras por política com hachura diferente.
  - **Eixo Y (secundário, marcadores ◆):** intensidade E/thr normalizada (Always-On = 1.000): **1.000 / 1.093 / 1.148 / 1.979**.
  - **Séries:** ΔE% (D1 watts) e Δthr% — 21 runs com IC95 (barras de erro); sobreposição do ΔE% do proxy (`baseline_summary_proxy.csv`: Random −52,0% no proxy vs. −19,0% em watts) como contorno pontilhado.
- **Anotações visuais:**
  - linha horizontal em **−66,5% / −33,5%** = teto físico de poupança por célula desligada no modelo EARTH (`(P_on−P_off)/P_on`, carga zero→máxima), rotulada "limite físico do sleep (EARTH)";
  - seta "economia ilusória" apontando o par Random (ΔE −19,0% vs. Δthr −59,0%);
  - faixa sombreada onde Δthr < ΔE (toda política cai nela).
- **Takeaway ("mensagem de venda"):** *toda poupança de energia observada é menor que a perda de throughput, e a energia por bit entregue piora em 100% das políticas — o "ganho energético" publicado é tráfego não entregue.*
- **Código → artefato:** dados de `analysis/baseline_summary.csv` + `_proxy.csv`, gerados por `analysis/recompute_baselines.py` (lê `examples/output/real_runs/manifest.json` + `database.db` de cada run via `examples/compare_runs.py::run_metrics`). Script de plot: **criar `analysis/plot_fig1_energy_illusion.py`** (matplotlib; lê os 2 CSVs).
- **Estado:** ✅ dados prontos (21 runs D1 + proxy); 🔄 atualizar para 40/40 após o rerun (checklist §4).

### Figura 2 — Anatomia do Reward Hacking na Eq. (2) (Massa + Sensibilidade)

- **Objetivo:** provar matematicamente que a Eq. (2) força o agente a desligar células — e que não é questão de calibrar pesos.
- **Formato:** dois painéis.
  - **(a) Stacked bar da massa negativa do reward** (2 barras: "pré-calibração (fallback)" vs. "pós D1–D3 (quantílica)"):
    - pré: `bs_on` **89,7%**, rlf 5,8%, δ 3,2%, energia 1,3% (contribuições −0,3002 / −0,0193 / −0,0109 / −0,0044; PPO-2 0.40/0.40/0.10/0.10);
    - pós: **energia 40% + `bs_on` 40%**, rlf 10%, δ 10% (−0,2000 cada) — evidenciando a **duplicação de w₂** (peso efetivo 0,80 no eixo energético: 0,40·energia + 0,40·bs_on);
    - **Eixo X:** regime de normalização; **Eixo Y:** % da massa negativa; segmentos rotulados com o termo da Eq. (2).
  - **(b) Heatmap política × agente da Tabela I** (6×4: PPO-1..5, DQN × Always-On/Static/Dynamic/random), célula sombreada pelo reward médio; ✱ no vencedor de cada linha:
    - **Random vence em 5/6** (PPO-2, PPO-3, PPO-4, PPO-5, DQN); Static vence só no PPO-1 (menor w₂=0,19);
    - reward médio por política: Always-On −0,466; Static −0,273; Dynamic −0,286; **Random −0,238**;
    - barra lateral com as correlações: `corr(R, bs_on) = −0,953`, `corr(R, thr) = −0,527`, `corr(thr, energy) = +0,781` (watts) — e, no regime proxy, `−0,978 / −0,438 / +0,893`.
- **Anotações visuais:** seta no painel (a) "a calibração corrige a escala, não a semântica"; no painel (b), retângulo vermelho na coluna Random com a legenda "pior política real (thr −59%, RLF máx 44) = ótimo do reward".
- **Takeaway:** *a Eq. (2) não codifica QoS: seu ótimo é desligar células (`bs_on`), e o peso do eixo energético é efetivamente 2·w₂ — nenhum ajuste linear de pesos conserta isso; é preciso uma restrição explícita.*
- **Código → artefato:** dados de `analysis/reward_decomposition.csv` + `_proxy.csv` + `reward_weights_sensitivity.csv` (+ `_proxy`), gerados por `analysis/reward_decomposition.py` (lê `examples/output/real_runs/baselines_dataset_d1d3.npz`, 2.079 transições, 21 episódios; calibração `reward_calibration_d1d3.npz`). Script de plot: **criar `analysis/plot_fig2_reward_hacking.py`**.
- **Estado:** ✅ dados prontos; 🔄 regenerar com dataset 40/40 (`--dataset` aponta o novo `.npz`).

### Figura 3 — A Fronteira de Pareto QoS-Aware (a Figura que Fecha o Artigo)

- **Objetivo:** posicionar a nossa solução (E1+E2: IQL/CQL com restrição de QoS) contra baselines reais e os pesos publicados da Tabela I, com o piso de SLA explícito.
- **Formato:** scatter 2D único (dupla coluna no paper: painel a com eixo Y = ΔThroughput%; painel b com eixo Y = RLF máximo).
  - **Eixo X:** Redução de Energia (%) vs. Always-On — modelo D1 watts.
  - **Eixo Y:** ΔThroughput (%) (painel a); RLF máximo por run (painel b).
  - **Séries/points:**
    - **Runs reais 34/40** (média por política/tupla, 5 seeds, barras de erro): Always-On (0,00 ± 0,13 / 0,00 ± 20,66), Random (**+21,39 ± 3,51 / −61,66 ± 9,57**), Static 4-2-0-1 (+6,09/−10,09), Static 3-2-0-2 (+12,05/−20,00), Static 2-2-0-3 (+18,22/−31,36), Dynamic 4-2-0-1 (+6,03/−11,90), Dynamic 3-2-0-2 (+12,15/−23,87), Dynamic 2-2-0-3 (+18,25/−34,92) — RLF máximo: Always-On 18, Random 48;
    - **Pontos publicados do artigo** ( marcadores ○ vazados, "valores do texto"): Static (+0,7/−23,5), Dynamic (+4,3/−27,6), PPO-1 (+1,9/−16,3), **PPO-2 (+24,2/−52,3)**, PPO-3 (+4,1/−27,5), PPO-4 (+19,3/−46,2), PPO-5 (+13,6/−34,0), DQN (+1,3/−9,6);
    - `[TARGET EXP: Aguardando treino IQL/CMDP]` — **IQL + CMDP-Lagrangiano** e **IQL + barreira**: esperado no quadrante **ΔE ≥ 10–20% com Δthr ≥ −10%** (a região hoje vazia), porque a restrição `thr ≥ τ_SLA` bloqueia as ações de colapso que dão reward; marcadores ★ com hallow.
- **Anotações visuais:**
  - **linha horizontal tracejada vermelha em Δthr = −10%** rotulada **"Piso mínimo de QoS (SLA: τ_SLA)"** (τ calibrado do `baselines_d1d3`: quantil inferior do throughput Always-On);
  - **Zona de Colapso de Tráfego (vermelha)** abaixo da linha — onde caem Random, PPO-2, PPO-4, PPO-5 e as tuplas 2-2-0-3;
  - **Zona de Operação Viável (verde)** acima da linha e com ΔE > 0 — hoje ocupada apenas marginalmente pelas tuplas 4-2-0-1 (que economizam pouco) e vazia na região ΔE 10–20%;
  - diagonal de referência 1:1 "1 pp de energia ↔ ~1,6–2,9 pp de vazão" (das nossas medidas).
- **Takeaway:** *nenhuma política existente — publicada ou re-simulada — entrega economia substancial acima do piso de QoS; a região viável está vazia à espera exatamente de um agente treinado sob a restrição que propomos.*
- **Código → artefato:** esboço já existe — `ns-o-ran-gym/examples/output/analysis/pareto_energy_throughput.png` + `pareto_points.csv` (agregação via `examples/compare_runs.py` sobre o `manifest.json` dos 40 jobs). Script definitivo: **consolidar em `analysis/plot_fig3_pareto.py`** (lê `pareto_points.csv`; adiciona painel RLF e os pontos `[TARGET EXP]`).
- **Estado:** ✅ esboço 34/40 pronto e conferido independentemente; 🔄 40/40 após rerun; ⏳ pontos E1/E2 após treino.

### Figura 4 — Impacto da Fidelidade do Modelo (Ablação D1–D4: Proxy vs. EARTH Watts + Normalização)

- **Objetivo:** mostrar que decisões metodológicas **invisíveis no texto** do artigo base (energia = contagem de TB; normalização em totais; replay como "PPO offline") **invertem o ranking** das políticas.
- **Formato:** matriz 2×2 de painéis (slope/dumbbell charts + heatmap).
  - **(a) Dumbbell proxy→watts por política** do ΔE%: Random **−52,01% → −18,96%**; Dynamic −15,91% → −10,49%; Static −14,06% → −11,38% (**Eixo X:** ΔE%; **Y:** política) — o proxy exagera a poupança ~2,7× na Random;
  - **(b) Slope da intensidade E/thr** proxy vs. watts: Random 1,172 → 1,979; Static 1,060 → 1,093; Dynamic 1,079 → 1,148 (**sempre > 1 nos dois modelos**);
  - **(c) Slope da massa negativa de `bs_on`** pré→pós calibração D3: **89,7% → 40%** (com a contrapartida energia 1,3% → 40%);
  - **(d) Heatmap do vencedor da Eq. (2)** por regime: proxy-fallback (**Random 6/6**) vs. D1–D3 calibrado (**Random 5/6, Static 1/6**) — a correção de escala move um voto, mas **não muda o vencedor do problema**: só a restrição de QoS muda.
- **Anotações visuais:** em (a), seta "ΔE do proxy ≈ Δthr (r=0,893)" vs. watts (r=0,781) — o proxy é *tráfego disfarçado de energia*; em (d), célula destacada do PPO-1 com a nota "menor w₂ = único caso em que o throughput sobrevive".
- **Takeaway:** *o ranking publicado é artefato de três escolhas metodológicas não declaradas; cada uma é testável (D1–D4 já implementadas como ablações com flag) e nenhuma delas muda a conclusão central — sem restrição de QoS, o ótimo do reward é destrutivo.*
- **Código → artefato:** `analysis/recompute_baselines.py --output-suffix _proxy` e `analysis/reward_decomposition.py` sobre `baselines_dataset.npz` (pré-calibração) vs. `baselines_dataset_d1d3.npz` (pós); sensibilidade: `reward_weights_sensitivity*.csv`. Script: **criar `analysis/plot_fig4_ablation.py`**.
- **Estado:** ✅ todos os dados prontos.

---

## 3. Tabelas Definitivas

### Tabela I — Auditoria Metodológica (Artigo Base vs. Implementação Auditada D1–D4)

| Eixo | Artigo Base (CCNC 2025) | Nossa Implementação Auditada (D1–D4) | Impacto na Validade |
|---|---|---|---|
| Energia | Eq. (1) `γ=EC·P_tx`; Fig. 3b reporta % (e W na literatura) | **D1:** EARTH macro `P_on=130+94·x` W, `P_sleep=75` W; proxy como ablação | Sem watts, "poupança" ∝ tráfego não entregue; com EARTH, teto físico de 33,5–66,5%/célula |
| Restrição de QoS | RLF/throughput só na **avaliação** | **E1:** CMDP no **treino** (Lagrangiano `r−λ(RLF−κ)` e barreira `thr≥τ_SLA`), `constraint_mode` implementado | Sem restrição, o ótimo do reward é desligar células (Random 5/6) |
| Algoritmo "offline" | "PPO offline" via SB3 | **D4:** IQL discreto (`nsoran/drl/iql.py`); replay legado como ablação | O replay ignora a ação do agente → behavior cloning ponderado; comparação PPO vs. DQN era inválida |
| Normalização | Não declara população | **D3:** quantílica por célula, calibrada em runs reais (`reward_calibration_d1d3.npz`) | Fallback comprime componentes (89,7% da massa em `bs_on`) e fabrica o hacking |
| Estado | "85 KPMs, 12 por célula" | Auditado: `mac_pdu_64qam` duplica `tb_64qam`; **SVD: rank(X₈₅)=70**; `STATE_VARIANTS["compact70"]` com posto completo | 15 dimensões colineares exatas; compacto melhora amostra-eficiência do IQL/CQL |
| δ (custo de ativação) | "activation cost", 71 µs wake-up | **D2:** denso (default) vs. impulso (ablação); RF/wake-up físico não modelado no ns-3 (`TurnOff` não desliga RF) | δ pune *estar ligado*, não acordar; wake-up declarado ausente |
| Dataset | ~3.000 sims / >300k pontos | 2.079 transições (21 runs) → Tier M: 400 runs / 39.600 (13,2% da escala), 520 h com 6 workers | Escala 0,7% hoje; offline RL exige campanha — orçada e planejada |
| Avaliação | "500 pontos" | 99 indicações/run × 5 seeds; **sempre live** (`--ns3-path`), replay só como diagnóstico | Replay não mede política (aviso no `eval_matrix.json`) |
| Estabilidade | Não reporta crashes | **34/40**: 6 crashes (3 `NS_ASSERT` buffer + 3 `SIGSEGV` TCP/SACK), 100% em políticas com células OFF; patch IHL aplicado | Desligar células sem restrição desestabiliza a pilha — achado científico próprio |

**Esboço LaTeX (IEEEtran):**

```latex
\begin{table*}[!t]
\caption{Methodological Audit of the CCNC~2025 Energy-Saving Formulation vs.\ Our Audited Implementation (D1--D4)}
\label{tab:audit}
\centering
\begin{tabular}{@{}p{1.9cm}p{2.6cm}p{3.4cm}p{4.2cm}@{}}
\toprule
\textbf{Axis} & \textbf{Base paper} & \textbf{Ours (D1--D4)} & \textbf{Validity impact} \\
\midrule
Energy & Eq.~(1) $\gamma{=}EC{\cdot}P_{tx}$ (proxy) & EARTH watts, $P_{on}{=}130{+}94x$, $P_{sleep}{=}75$ & proxy savings $\propto$ undelivered traffic \\
QoS constraint & evaluation only & CMDP in training (Lagrangian/barrier) & unconstrained optimum is destructive \\
Offline RL & ``PPO offline'' (SB3 replay) & discrete IQL; replay as ablation & replay ignores the action $\Rightarrow$ weighted BC \\
Normalization & undeclared & per-cell quantiles, real-run calibration & fallback puts 89.7\% of mass on $B_{sON}$ \\
State & ``85 KPMs'' & $\mathrm{rank}(X_{85}){=}70$; compact70 & 15 exactly collinear dims \\
$\delta$ & ``activation cost'' & dense (D2) vs.\ impulse; no RF model & penalizes being ON, not waking up \\
Dataset & $\sim$3k sims / 300k pts & 2{,}079 $\rightarrow$ 39{,}600 transitions (Tier~M) & 0.7\% today; campaign budgeted \\
Stability & not reported & 6/40 crashes, all in sleep policies & unrestricted sleep destabilizes the stack \\
\bottomrule
\end{tabular}
\end{table*}
```

### Tabela II — Resultados Empíricos dos Baselines e Diagnóstico

Runs reais ns-3, 10 s, 99 indicações, modelo D1 (watts). IC95 = intervalo de confiança 95% (t). Ranking Eq. (2) sob pesos PPO-2/DQN (idem nos 5/6; PPO-1 prefere Static).

**Bloco A — 21 runs (oficial, com IC95):**

| Política | Runs | Throughput (média ± IC95) | Energia (W, ± IC95) | Δthr (%) | ΔE (%) | E/thr vs Always-On | Status no ranking Eq. (2) |
|---|---|---|---|---|---|---|---|
| Always-On | 3 | 76.686 ± 39.613 | 919,5 ± 3,7 | 0,00 | 0,00 | **1,000** | 4.º (pior; reward −0,466) |
| Static | 8 | 62.196 ± 7.733 | 814,8 ± 38,9 | −18,90 | −11,38 | 1,093 | 2.º (−0,273); 1.º sob PPO-1 |
| Dynamic | 7 | 59.769 ± 9.040 | 823,0 ± 39,2 | −22,06 | −10,49 | 1,148 | 3.º (−0,286) |
| Random | 3 | 31.407 ± 21.291 | 745,1 ± 30,2 | −59,04 | −18,96 | **1,979** | **1.º (melhor reward: −0,238)** |

**Bloco B — campanha Tier S 34/40 (preliminar, por tupla; ΔE+ = redução):**

| Política (tupla) | Seeds ok | ΔE (%) | Δthr (%) | RLF médio (máx) |
|---|---|---|---|---|
| Always-On | 5/5 | +0,00 ± 0,13 | 0,00 ± 20,66 | 7,59 (18) |
| Static 4-2-0-1 | 3/5 ⚠ | +6,09 ± 0,06 | −10,09 ± 12,34 | 9,83 (18) |
| Static 3-2-0-2 | 5/5 | +12,05 ± 0,05 | −20,00 ± 14,51 | 11,79 (20) |
| Static 2-2-0-3 | 4/5 ⚠ | +18,22 ± 0,09 | −31,36 ± 18,90 | 17,31 (26) |
| Dynamic 4-2-0-1 | 4/5 ⚠ | +6,03 ± 0,19 | −11,90 ± 9,98 | 8,99 (18) |
| Dynamic 3-2-0-2 | 5/5 | +12,15 ± 0,11 | −23,87 ± 12,90 | 12,30 (22) |
| Dynamic 2-2-0-3 | 3/5 ⚠ | +18,25 ± 0,06 | −34,92 ± 14,27 | 16,75 (27) |
| Random | 5/5 | +21,39 ± 3,51 | −61,66 ± 9,57 | 18,79 (48) |

⚠ = seeds excluídas por crash ns-3 (ver §4). **Blocos futuros:** `40/40 (pós-rerun)` — `[TARGET EXP: Aguardando rerun dos 6 crashes]`; `IQL+CMDP` — `[TARGET EXP: Aguardando treino IQL/CMDP]`, colunas iguais às do Bloco B.

**Esboço LaTeX (bloco principal):**

```latex
\begin{table}[!t]
\caption{Real ns-3 Baselines (21 runs, D1 watts): ``Energy Savings'' Are Undelivered Traffic}
\label{tab:baselines}
\centering
\begin{tabular}{@{}lrrrrc@{}}
\toprule
Policy & Thr. & $\Delta$thr & $\Delta$E & $E/\text{thr}$ & Eq.~(2) \\
 & (mean$\pm$IC95) & (\%) & (\%) & ratio & rank \\
\midrule
Always-On & $76.7k{\pm}39.6k$ & 0 & 0 & 1.000 & 4th \\
Static    & $62.2k{\pm}7.7k$  & $-18.9$ & $-11.4$ & 1.093 & 2nd \\
Dynamic   & $59.8k{\pm}9.0k$  & $-22.1$ & $-10.5$ & 1.148 & 3rd \\
Random    & $31.4k{\pm}21.3k$ & $-59.0$ & $-19.0$ & \textbf{1.979} & \textbf{1st} \\
\midrule
\multicolumn{6}{l}{\footnotesize [TARGET EXP: 40/40 rerun + IQL/CMDP rows]} \\
\bottomrule
\end{tabular}
\end{table}
```

### Tabela III — Matriz Experimental de Estado da Arte (E1 + E2)

Colunas de resultado: ΔE% (redução, +), Δthr%, RLF médio (máx), violações do SLA por episódio (`thr < τ_SLA`), distância à behavior policy (TV) / ações OOD. Métrica primária pré-registrada: **ΔE vs. Always-On com Δthr ≥ −10% e RLF ≤ 2 UEs** (critério E1 do `research-extensions.md`).

| Grupo | Política / Agente | ΔE% (+) | Δthr% | RLF (máx) | Violações SLA/ep | Status |
|---|---|---|---|---|---|---|
| **Baselines** | Always-On | 0,00 | 0,00 | 7,59 (18) | 0 | ✅ 34/40 medido |
| | Static 3-2-0-2 | +12,05 | −20,00 | 11,79 (20) | > 0 | ✅ medido |
| | Dynamic 3-2-0-2 | +12,15 | −23,87 | 12,30 (22) | > 0 | ✅ medido |
| | Random | +21,39 | −61,66 | 18,79 (48) | > 0 | ✅ medido |
| **Artigo base** (publicado) | PPO-2 (Eq. 2) | +24,2 | −52,3 | máx 2 (relatado) | n/d | ○ valores do texto |
| | DQN (REM+CQL) | +1,3 | −9,6 | máx 1 (relatado) | n/d | ○ valores do texto |
| **E2: sem restrição** | BC (replay ponderado) | `[TARGET EXP: Aguardando treino IQL/CMDP]` | | | | espelha a behavior policy (F2) |
| | CQL (REM-DQN, D4) | `[TARGET EXP]` | | | | espera-se: ganho sem QoS, colapso parcial |
| | IQL puro (D4) | `[TARGET EXP]` | | | | espera-se: domina BC; sem garantia de SLA |
| **E1: com restrição (nosso)** | IQL + CMDP-Lagrangiano ($r{-}\lambda(\text{RLF}{-}\kappa)$, λ∈{0..10}) | `[TARGET EXP]` | **≥ −10** | **≤ 2** | **→ 0** | região viável do Pareto (Fig. 3) |
| | IQL + barreira ($thr \ge \tau_{SLA}$) | `[TARGET EXP]` | **≥ −10** | **≤ 2** | **→ 0** | robustez à formulação |

Comportamento esperado dos `[TARGET EXP]` (pré-registrado, `research-extensions.md` §E1/E2): (i) BC reproduz a behavior policy e não a supera; (ii) CQL/IQL sem restrição convergem para gain de energia com violação de QoS (região vermelha da Fig. 3); (iii) **IQL/CQL + CMDP é o único braço esperado na zona verde** — se nenhuma política cumprir κ, o resultado reportado é a própria fronteira de inviabilidade (resultado negativo útil, declarado no plano).

**Esboço LaTeX:** mesma estrutura de `tabular` da Tabela II com colunas `Policy & $\Delta$E(\%)$ & $\Delta$thr(\%)$ & RLF & SLA viol. & Status`, grupo separado por `\midrule` e marcadores `\textcolor{red}{...}/\textcolor{green!60!black}{...}` por zona.

---

## 4. Triagem de Rigor Experimental (Status 34/40 e Próximos Passos)

### 4.1 Quadro da campanha Tier S

| Item | Valor | Fonte |
|---|---|---|
| Jobs planejados | 40 (8 políticas × 5 seeds 555–559) | `experiment-configs/baselines_5seeds.json` |
| Completos e comparáveis | **34** | `manifest.json` + `pareto_points.csv` |
| Crashes | **6** (3 `NS_ASSERT` de buffer + 3 `SIGSEGV`) | `docs/plano_tecnico/RELATORIO_PLANO_TECNICO.md` |
| Distribuição dos crashes | 100% em políticas com células OFF; Always-On 5/5 ok; tuplas agressivas 2-2-0-3 concentram 4/6 | triagem por run |
| Causas (confirmadas por coredump) | 2× assert de escrita `buffer.h:883` via `Ipv4Header::Serialize` (IHL=0, t=7,639932242 s) — **patch aplicado e compilando**; 1× assert de leitura (família IPv6) — **fix de 28/out cobre** (repro passa do ponto de morte); 3× SIGSEGV `TcpTxBuffer::Update ← ProcessOptionSack` (t≈9,9 s, com fixes SACK presentes) — **risco residual** | workflows de triagem |
| Independência amostral | `dynamic_2_2_0_3_seed556` e `static_2_2_0_3_seed556` têm fingerprints idênticos (mesma trajetória antes do crash) — **não contar como 2 amostras** | `implementation-audit.md` §4 |

### 4.2 Decisão de rigor: rerun (principal) + análise de falha (complementar)

1. **Rerun dos 6 com o build corrigido** é o caminho principal: o patch IHL (`ipv4-header.cc:444-448`) e o fix IPv6 fecham deterministicamente os 3 asserts; os 3 SIGSEGVs TCP/SACK são re-executados e, se recorrerem, capturados sob gdb (`NS3_LAUNCH_PREFIX="gdb -batch -ex run -ex 'bt 60' --args"`) — **nenhum segundo patch antes de backtrace novo**. Isso fecha a tabela simétrica 40/40 = 8 × 5 que revisores esperam.
2. **Análise de correlação de falha** entra como resultado, não como desculpa: falha × agressividade da política (todas as 6 em políticas com sleep; 0/5 em Always-On; concentração nas tuplas com 3 células OFF) — é o **achado de estabilidade** do artigo: o espaço de ações irrestrito não é operacionalmente viável; métricas de estabilidade devem acompanhar energia/throughput/RLF (parágrafo pronto em `docs/plano_tecnico/RELATORIO_PLANO_TECNICO.md` §1.6).
3. **Protocolo estatístico:** IC95 bootstrap sobre 5 seeds por política; médias por tupla reportadas separadamente (a agregação por política esconde o efeito da tupla); crashes **reportados, nunca descartados em silêncio** (invariante do `decisions.md`); pré-registro de hipóteses antes do treino E1/E2 (extensão E7).

### 4.3 Checklist de Execução Rápida (comandos exatos, verificados)

```bash
# (1) RERUN DOS 6 CRASHES com o build já corrigido (patch IHL + IPv6)  [~6×90 min, 6 workers]
cd /mnt/legitbet/elioth_sbrc/ns-o-ran-gym
.venv/bin/python scripts/run_paper_baselines.py \
  --config src/environments/scenario_configurations/paper_es_use_case.json \
  --ns3-path ../ns-3-mmwave-oran --output-dir examples/output/real_runs \
  --seeds 555 556 557 558 559 --jobs 6 --force \
  --only dynamic_2_2_0_3_seed555 dynamic_2_2_0_3_seed556 static_2_2_0_3_seed556 \
         static_4_2_0_1_seed558 static_4_2_0_1_seed559 dynamic_4_2_0_1_seed558
#    → se algum SIGSEGV recorrer: NS3_LAUNCH_PREFIX="gdb -batch -ex run -ex 'bt 60' --args" no job

# (2) ATUALIZAR AS TABELAS/FIGURAS DE 21 → 40/40 runs  (regenera baseline_summary*.csv|md, por tupla)
cd /mnt/legitbet/elioth_sbrc
ns-o-ran-gym/.venv/bin/python analysis/recompute_baselines.py            # modelo D1 (watts)
ns-o-ran-gym/.venv/bin/python analysis/recompute_baselines.py --output-suffix _proxy  # ablação Fig. 4
ns-o-ran-gym/.venv/bin/python analysis/reward_decomposition.py \
  --dataset ns-o-ran-gym/examples/output/real_runs/baselines_dataset_d1d3.npz         # Fig. 2

# (3) RECONVERTER O DATASET offline com as runs completas (34→40) antes do treino
cd /mnt/legitbet/elioth_sbrc/ns-o-ran-gym
.venv/bin/python examples/collect_real_dataset.py --runs-dir examples/output/real_runs \
  --agent PPO-2 --output examples/output/real_runs/baselines_dataset_40runs.npz

# (4) PRIMEIRO TREINO E2 (IQL, D4) + braço E1 (CMDP)
.venv/bin/python examples/ppo_2.py --dataset examples/output/real_runs/baselines_dataset_40runs.npz \
  --offline-algo iql --total-timesteps 1200000 --seed 0 \
  --output-dir examples/output/training/PPO-2_iql_s0
#    E1/CMDP: constraint_mode ("lagrangian"|"barrier", com tau_sla/epsilon) já existe em
#    src/nsoran/drl/config.py (design: docs/plano_tecnico/cmdp_design.md, grade λ∈{0..10});
#    falta expor --constraint-mode/--tau-sla no examples/_agent_cli.py (~30-60 min) e treinar
#    os 2 braços E1 nas mesmas seeds — [TARGET EXP] da Tabela III e da Fig. 3.
```

**Sequência crítica:** (1) → (2) regenera Figs. 1, 2 e 4 com 40/40 → (3) → (4) preenche os `[TARGET EXP]` da Fig. 3 e da Tabela III. Orçamento: rerun ≈ 1,5 dia de parede (6 workers); treinos E1/E2 conforme `training_matrix.json` (18 + ablações); avaliações live conforme `eval_matrix.json` (aviso: sem `--ns3-path` a avaliação **não** mede política).
