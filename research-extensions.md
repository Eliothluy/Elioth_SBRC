# Extensões de pesquisa — candidatas e priorização

Este documento propõe **extensões** ao trabalho de Bordin et al. (CCNC 2025),
fundamentadas (i) nas divergências encontradas na auditoria do código
(`implementation-audit.md`, §3) e (ii) nos 27 trabalhos validados da matriz
`related-work-matrix.csv`. Nenhuma extensão aqui é um compromisso: são candidatas
com pergunta, hipótese, desenho, custo e risco, para escolha antes de qualquer
alteração de código.

> **Aviso de novidade.** As afirmações de novidade são *provisórias*: resultam da
> matriz de 27 trabalhos, não de uma revisão sistemática dedicada a cada tópico.
> Antes de comprometer uma extensão, fazer uma busca dirigida (OpenAlex + Crossref)
> e registrar em `literature-search-log.md`, como na Fase F.

## 1. O que a auditoria deixou em aberto

| ID | Achado | Evidência | Consequência científica |
|---|---|---|---|
| **F1** | O reward é dominado por `bs_on` (89,7% da massa negativa) e o **Random** é o óptimo do reward em todos os seis pesos da Tabela I | `analysis/reward_decomposition.md` | A função objectivo não codifica QoS; treinar com ela não pode, por si só, reproduzir o compromisso energia–throughput do artigo |
| **F2** | O PPO offline usa `OfflineReplayEnv`, que ignora a acção do agente | `implementation-audit.md` §3.5; `offline.py:659–705` | O “PPO” é behavior cloning ponderado pelo reward, não offline RL; a comparação PPO vs DQN do artigo não é reproduzível neste código |
| **F3** | “Energia” é `TB.TotNbrDl.1 × 1.0`, sem watts nem potência estática | `implementation-audit.md` §3.1; `es_env.py` | A poupança de 24,2% não é comparável com a Fig. 3b (W) nem com a literatura EARTH/E3F |
| **F4** | $\delta = 0.9^{0.01 T_{on}}$ é custo **denso enquanto ligado**, não impulso OFF→ON; a Fig. 3d mostra valores > 1 | `mathematical-formulation.md` R2 | O termo de “activation cost” pune manter células ligadas, não acordá-las; a semântica do artigo não é a do código |
| **F5** | `TurnOff` só muda o flag de handover, expulsa UEs e faz `SetCellState(OFF)`; não há desligamento de RF nem atraso de 71 µs | `implementation-audit.md` §3.1; `mmwave-enb-net-device.cc` | O simulador não modela o custo físico que justifica o problema; a política aprende a evitar tráfego, não a dormir |
| **F6** | A normalização é aplicada a totais da rede, não por célula | `implementation-audit.md` §3.4 | O reward depende do número de células ligadas de forma não linear; a escala muda com N |
| **F7** | Só 21 runs completos, 3 seeds por baseline, 3 crashes; Always-On varia ±20% entre seeds | `analysis/baseline_summary.md` | Qualquer efeito < ~20% não é distinguível de ruído; a Tabela VI não é reproduzível com esta amostra |
| **F8** | O artigo reporta 500 pontos de avaliação; o cenário local tem 99 indicações por run de 10 s | `article-understanding.md` §“Inconsistência I3”; `reproduction-plan.md` | Falta potência estatística e horizonte; políticas de sleep precisam de escalas de segundos a minutos |
| **F9** | `mac_pdu_64qam` duplica `tb_64qam`; 85 dimensões transportam 83 | `implementation-audit.md` §3.2 | Estado redundante; a ablation do estado é barata e informativa |
| **F10** | A energia e o throughput têm `r = 0.893`; a intensidade E/thr **piora** em todas as políticas | `analysis/reward_decomposition.md`; `analysis/baseline_summary.md` | “Poupar energia” é quase equivalente a “entregar menos tráfego”; a métrica do artigo não separa as duas coisas |

## 2. Critérios de seleção

Uma extensão só entra no núcleo da dissertação se cumprir **todos**:

1. **Dados reais** — ns-3 via Gym, sem `--synthetic`, sem replay como avaliação.
2. **Mensurável** — métrica primária definida antes da campanha, com IC e seeds.
3. **Executível** — custo marginal ≤ ~2 semanas de parede na máquina disponível
   (20 cores, 15 GiB RAM, 207 GB livres), já incluindo o Tier M.
4. **Novidade defensável** — posicionada contra pelo menos 2 trabalhos da matriz.
5. **Encaixe na tese** — produz um capítulo ou uma secção com figura/tabela própria.

Ficam **fora** do núcleo: extensões que exigem testbed (OpenRAN Gym/ColO-RAN),
novo simulador, ou mais de ~150 runs adicionais.

## 3. Catálogo de extensões

### E1 — Reward com restrição de QoS (constrained MDP)

- **Pergunta.** A Eq. (2) com uma restrição explícita de QoS produz políticas que
  dominam Static/Dynamic na fronteira energia–throughput–RLF?
- **Hipótese.** Sim: com $\mathbb{E}[\text{RLF}] \le \kappa$ e um piso de throughput,
  o agente deixa de ganhar reward desligando células e passa a ganhar por desligar
  **sem** violar o serviço; o Random deixa de ser óptimo.
- **Motivação.** F1, F10; o artigo usa RLF como critério de avaliação mas não como
  restrição de treino. A literatura de sleep com QoS (Liu 2016; masked offline RL,
  ICCC 2025) mostra que a restrição muda a política, mas nenhum trabalho o faz com
  os 85 KPMs e o controlo E2 do O-RAN.
- **Desenho.** Quatro braços, mesma dataset e mesmos algoritmos:
  1. Eq. (2) original (controle);
  2. Eq. (2) + penalidade quadrática em $\max(0, \text{RLF} - \kappa)$;
  3. CMDP com Lagrangiano: $r_\lambda = r - \lambda(\text{RLF} - \kappa)$, $\lambda$ atualizado por descida de gradiente;
  4. Piso de throughput: $r$ inalterado, acção mascarada se o throughput previsto < $T_{\min}$ (modelo de transição simples ou máscara por célula).
- **Métricas.** Δenergia vs Always-On **a** Δthroughput ≥ −10% **e** RLF ≤ 2 UEs; área sob a fronteira (hipervolume); violações de restrição por episódio; recompensa média.
- **Custo.** 4 braços × 5 seeds = 20 treinos offline (~20 h CPU) + 20 avaliações live
  (~4,3 h de parede com 6 workers). Reutiliza o dataset do Tier M.
- **Dependências.** D4 (algoritmo offline válido); calibração do reward (F1).
- **Risco.** A restrição pode tornar o problema inviável (nenhuma política cumpre
  κ); nesse caso o resultado é a própria fronteira de inviabilidade — reportar.
- **Novidade.** O artigo não tem restrição; o masked offline RL (2025) tem QoS mas
  não O-RAN/ns-3; os trabalhos de sleep com QoS não usam 85 KPMs.
- **Capítulo.** “Reward design para energy saving com QoS em O-RAN”.

### E2 — Comparação de algoritmos offline (BC vs CQL vs IQL vs TD3+BC vs BCQ)

- **Pergunta.** Qual é o custo, em desempenho live, de tratar o cell sleep como
  offline RL em vez de behavior cloning?
- **Hipótese.** CQL/IQL superam BC (PPO-replay) sobretudo quando a behavior policy
  é fraca (Random/Static), porque penalizam acções fora da distribuição; a diferença
  desaparece quando o dataset é dominado por Always-On.
- **Motivação.** F2; o artigo diz “offline” mas o PPO implementado não é offline RL;
  a literatura (TWC 2024; survey TNNLS 2024; IQL/TD3+BC/BCQ) mostra que o algoritmo
  importa e que o desempenho depende da behavior policy.
- **Desenho.** Fatorial 2 (reward: Eq. 2 vs E1-CMDP) × 4 algoritmos (BC, CQL, IQL,
  TD3+BC; BCQ como opcional) × 5 seeds = 40 treinos; 40 avaliações live.
- **Métricas.** Reward offline; taxa de acções OOD; Δenergia/Δthroughput/RLF live;
  distância entre a política aprendida e a behavior policy (TV distance por estado).
- **Custo.** ~40 treinos (0,5–2 h cada) + 40 runs (~8,7 h) ≈ 3–4 dias.
- **Dependências.** D4; dataset do Tier M; implementação de IQL/TD3+BC (bibliotecas
  ou ~300 linhas).
- **Risco.** TD3+BC/BCQ são para acções contínuas; a adaptação a `Discrete(128)`
  exige cuidado (Gumbel/one-hot ou CQL discreto). IQL é discreto e mais simples.
- **Novidade.** Primeira comparação sistemática de offline RL para cell sleep em
  O-RAN com estado de 85 KPMs e avaliação live em ns-3.
- **Capítulo.** “Offline RL para cell sleep: algoritmos e behavior policies”.

### E3 — Fidelidade do modelo de energia (proxy → watts)

- **Pergunta.** A ordenação de políticas e a poupança reportada mudam quando a
  energia é modelada em watts (EARTH/E3F) em vez do proxy `TB × 1.0`?
- **Hipótese.** Sim, e de forma assimétrica: políticas agressivas parecem melhores
  no proxy porque o proxy é proporcional ao tráfego; com potência estática, o
  benefício de desligar células aumenta, mas o custo de acordar e o consumo de
  idle reduzem a poupança.
- **Motivação.** F3; a Fig. 3b do artigo reporta W, a Eq. (1) reporta PDU·W; a
  literatura EARTH/E3F e os surveys de EE em 5G/O-RAN fornecem o modelo.
- **Desenho.** (i) instrumentar ns-3 para expor `P_0`, `Δp`, PRB e estado RF por
  célula; (ii) duas variantes: `P = P_0 + Δp·PRB` (sem sleep) e `P = P_sleep` quando
  OFF; (iii) recalcular as 21 runs existentes **sem retreinar** (a energia não entra
  no estado, só no reward); (iv) retreinar o melhor agente de E2 com o novo reward.
- **Métricas.** Δenergia em W vs proxy; correlação de rankings (Spearman); energia
  por bit; contribuição de `P_0` vs `Δp·PRB`.
- **Custo.** Desenvolvimento ns-3 2–3 dias; recálculo offline ~1 h; 10 runs de
  validação (~2,2 h); retreino 5 seeds (~5 h). Total ~1 semana.
- **Dependências.** D1; acesso ao `mmwave-enb-net-device` e ao PHY para PRB.
- **Risco.** O modelo EARTH foi calibrado para LTE macro; os parâmetros para mmWave
  FR1 são uma escolha — documentar como hipótese, não como verdade.
- **Novidade.** Quantifica o erro de usar um proxy de tráfego como energia num
  resultado publicado de O-RAN; não encontrei trabalho que o faça.
- **Capítulo.** “Que energia estamos a poupar? Fidelidade do modelo”.

### E4 — Semântica do custo de activação e wake-up

- **Pergunta.** Custo denso vs impulso OFF→ON muda a política de sleep?
- **Hipótese.** O impulso penaliza o *chattering* (ligar/desligar frequente) e
  produz políticas com menos transições e maior tempo médio em sleep; o custo denso
  atual favorece manter ligado, o oposto do objectivo.
- **Motivação.** F4; o artigo fala em “activation cost” e 71 µs de wake-up, mas o
  código implementa `0.9^{0.01 T_on}`; a literatura de advanced sleep modes modela
  níveis e timers.
- **Desenho.** Três variantes: (a) denso atual; (b) impulso $\delta = c$ na transição
  0→1, 0 caso contrário; (c) impulso + bloqueio de 71 µs + histerese mínima de 100 ms.
  Medir transições por run, tempo em sleep, energia e throughput.
- **Custo.** 3 variantes × 5 seeds = 15 treinos + 15 runs ≈ 1,5 dias, mais 2 dias de
  código ns-3/Python.
- **Dependências.** D2; F5 (se o RF não desliga, o wake-up é cosmético).
- **Risco.** O bloqueio de 71 µs é muito menor que a indicação de 100 ms; o efeito
  pode ser nulo — nesse caso o resultado é negativo e útil (mostra que a escala do
  artigo não é a do simulador).
- **Novidade.** Nenhum trabalho de O-RAN com 85 KPMs compara as duas semânticas.
- **Capítulo.** Secção do capítulo de modelação.

### E5 — Ablation do estado (redundância e normalização por célula)

- **Pergunta.** As 85 dimensões são todas necessárias? A duplicação
  `mac_pdu_64qam = tb_64qam` e a normalização global prejudicam o aprendizado?
- **Hipótese.** Remover a duplicata e normalizar por célula melhora a amostra
  eficiência e a transferência entre N de células ligadas.
- **Motivação.** F6, F9; o artigo enumera 10 KPMs mas usa 12 por célula.
- **Desenho.** 4 variantes de estado: (a) 85 completo; (b) 78 (remove duplicata);
  (c) 85 com normalização por célula; (d) 78 + por célula. Mesmo algoritmo (CQL),
  5 seeds.
- **Custo.** 4 × 5 = 20 treinos + 20 runs ≈ 2 dias.
- **Dependências.** D3; dataset do Tier M.
- **Risco.** Efeito pequeno face à variância de seed; precisa de 5 seeds e IC.
- **Novidade.** Ablation de KPMs é rara na literatura de ES em O-RAN.
- **Capítulo.** Secção de representação de estado.

### E6 — Calibração e população da normalização

- **Pergunta.** A normalização de fallback comprime as componentes em ≈0,005 e
  torna o reward dominado por `bs_on`; uma calibração nos quantis reais corrige isso?
- **Hipótese.** Sim: a calibração reduz a dominância de `bs_on` e faz o ranking de
  políticas coincidir com a métrica de avaliação (energia/throughput/RLF).
- **Motivação.** F1; `examples/fit_reward_calibration.py` existe mas não foi usado
  no dataset atual; a Eq. (2) pede quantis da distribuição de referência.
- **Desenho.** Comparar três normalizações: fallback, quantis por componente
  (uniforme) e quantis por célula; medir share de `bs_on`, correlação com throughput
  e ranking das 4 baselines.
- **Custo.** Sem runs novos: recalcular o dataset e reavaliar 3 × 5 treinos + 15 runs
  ≈ 1,5 dias.
- **Dependências.** D3.
- **Risco.** A calibração é sensível à behavior policy do dataset; documentar a
  população usada (só Always-On? todos os baselines?).
- **Novidade.** Baixa; é sobretudo rigor metodológico. Pode ser uma subsecção de E1.
- **Capítulo.** Metodologia.

### E7 — Protocolo estatístico e horizonte de avaliação

- **Pergunta.** Com quantas seeds e que horizonte se distingue uma política de sleep
  do ruído? A conclusão do artigo sobrevive a um protocolo pré-registado?
- **Hipótese.** O efeito de 24,2% é detectável com 5 seeds e 99 indicações, mas as
  diferenças entre PPO-1/PPO-4/PPO-5 (13–19%) não são; horizontes de 30 s reduzem a
  variância e revelam *chattering* que 10 s esconde.
- **Motivação.** F7, F8; 3 seeds e 10 s são o mínimo, não o suficiente.
- **Desenho.** (i) análise de potência sobre os 21 runs existentes (bootstrap);
  (ii) 10 runs de 30 s (Always-On, Static, Dynamic, Random × 2–3 seeds);
  (iii) pré-registo de hipóteses e critérios de aceitação antes da campanha.
- **Custo.** 10 runs de 30 s ≈ 39 h de parede (6,5 h com 6 workers) + análise 1 dia.
- **Dependências.** nenhuma além do Tier M.
- **Risco.** 30 s pode expor crashes residuais (3/24 em 10 s) → orçar retrabalho.
- **Novidade.** Estudo de reprodutibilidade com ns-3 real; publicável como *negative
  result* / workshop.
- **Capítulo.** Capítulo de metodologia e validade.

### E8 — Coordenação com outras xApps (sleep + load balancing/power)

- **Pergunta.** O cell sleep entra em conflito com handover/load balancing? Uma
  máscara de acção ou um gestor de conflito evita oscilações?
- **Hipótese.** Sem coordenação, o sleep aumenta handovers e RLF; com máscara
  (não desligar a única célula que serve UEs em risco) o RLF cai sem perder poupança.
- **Motivação.** COMIX (2025) e Explainable/Conflict-Aware (2026); o artigo não
  trata conflitos, mas o cenário tem 7 células e handover dinâmico.
- **Custo.** 2 braços × 5 seeds = 10 runs + 5 dias de desenvolvimento ≈ 2 semanas.
- **Dependências.** E1 (máscara), F5.
- **Risco.** Escopo grande; provavelmente *stretch*.
- **Novidade.** Coordenação explícita de sleep com handover em O-RAN/ns-3.
- **Capítulo.** Trabalho futuro ou capítulo de extensão.

### E9 — Política interpretável (destilação)

- **Pergunta.** É possível destilar a política DRL numa árvore/NAM monotónica sem
  perder poupança?
- **Hipótese.** Sim, com perda < 2 p.p. de energia, e a política destilada revela
  regras do tipo “desligar célula i se PRB < x e RLF < y”.
- **Motivação.** Explainable/Conflict-Aware (2026); a opacidade é uma barreira à
  adopção de xApps.
- **Custo.** 3 dias de código + 10 runs de avaliação ≈ 1 semana.
- **Dependências.** E1/E2 (política treinada).
- **Risco.** Pode não haver dados suficientes para destilar 128 acções.
- **Novidade.** Média; há trabalho recente em NAM para cell sleep.
- **Capítulo.** Trabalho futuro.

### E10 — Generalização entre misturas de tráfego e topologias

- **Pergunta.** Um agente treinado com `trafficModel=4` mantém a poupança com
  `trafficModel=3` ou outro número de UEs?
- **Hipótese.** Não: a política aprende atalhos específicos da mistura (25% full
  buffer) e degrada quando a fracção de UDP/bursty muda.
- **Motivação.** O artigo usa uma única mistura; a literatura de offline RL com
  mixture datasets mostra dependência da behavior policy.
- **Custo.** 2 misturas novas × 40 runs de dataset = 80 runs (~17 h) + 10 treinos +
  10 avaliações ≈ 4 dias.
- **Dependências.** Tier M; E2.
- **Risco.** Custo alto; só depois do núcleo.
- **Novidade.** Generalização cross-traffic em ES/O-RAN com ns-3.
- **Capítulo.** Trabalho futuro.

## 4. Priorização

Score = 0,5·impacto + 0,3·viabilidade + 0,2·custo (5 = melhor).

| ID | Impacto | Viabilidade | Custo (5 = barato) | Score | Papel |
|---|---|---|---|---|---|
| E2 | 5 | 5 | 4 | **4,8** | núcleo |
| E1 | 5 | 4 | 4 | **4,5** | núcleo |
| E7 | 4 | 5 | 4 | **4,3** | núcleo |
| E6 | 3 | 5 | 4 | 3,8 | núcleo (subsecção de E1) |
| E5 | 3 | 5 | 3 | 3,6 | secundária |
| E3 | 4 | 3 | 3 | 3,5 | secundária |
| E4 | 3 | 3 | 3 | 3,0 | secundária |
| E9 | 3 | 3 | 3 | 3,0 | stretch |
| E8 | 4 | 2 | 2 | 2,9 | stretch |
| E10 | 4 | 2 | 1 | 2,5 | futuro |

**Núcleo recomendado (Tier M + ~1 semana):**

1. **E2** — sem algoritmo offline válido, nenhum resultado de agente é interpretável.
2. **E1 + E6** — reward com QoS e normalização calibrada; resolve F1 e dá sentido à
   comparação.
3. **E7** — protocolo estatístico e horizonte; protege todos os resultados.
4. **E3** (se houver tempo) — liga o resultado à literatura de energia e responde à
   objeção “isto não é energia”.

**Secundárias:** E5 (ablation barata, bom capítulo de estado), E4 (semântica de δ).
**Stretch:** E8, E9, E10.

## 5. O que **não** é extensão

- Repetir os 21 baselines com mais seeds → é reprodução (Tier M), não novidade.
- Trocar PPO por CQL sem desenho experimental → é E2, mas sem hipótese não é tese.
- Aumentar o `simTime` sem medir o efeito → é E7.
- Treinar com dados sintéticos → proibido pela política de dados do projeto.

## 6. Ameaças à validade

| Ameaça | Efeito | Salvaguarda |
|---|---|---|
| Variância de seed (Always-On ±20%) | falsos positivos | 5 seeds, IC bootstrap, pré-registo |
| Crashes residuais (3/24) | perda de runs, viés de sobrevivência | corrigir antes; reportar runs perdidos; repor seeds |
| Reward dominado por `bs_on` | políticas degeneradas | E1/E6; critério de aceitação “< 80%” |
| Energia = proxy | conclusões não transferíveis | E3; reportar as duas métricas |
| Dataset de uma só mistura | generalização desconhecida | E10 como trabalho futuro; declarar limite |
| Offline replay | avaliação inválida | E2; avaliar sempre live |
| Overfitting ao cenário | novidade fraca | pelo menos 2 topologias/misturas em E10 |

## 7. Posicionamento e publicação

- **E2 + E1** dão um artigo de conferência (workshop/conferência IEEE) sobre
  “offline RL com restrição de QoS para cell sleep em O-RAN”, com ns-3 real.
- **E7 + E3** dão um artigo de reprodutibilidade/avaliação (“quanto do ganho é
  energia e quanto é tráfego?”), adequado a workshop de reprodutibilidade.
- **E5 + E6** são material de dissertação, não de artigo autónomo.
- **E8/E9/E10** ficam como trabalho futuro declarado, com desenho esboçado.

## 8. Mapa extensão → roadmap

| Fase do roadmap | Extensões | Portão |
|---|---|---|
| Fase 2 (reprodução) | — | 40 baselines, 5 seeds |
| Fase 3 (dataset) | — | ≥380 runs, calibração |
| Fase 4 (núcleo) | E2, E6 | algoritmo válido, reward não degenerado |
| Fase 5 (núcleo) | E1, E7 | restrição de QoS, 5 seeds |
| Fase 6 (secundárias) | E3, E4, E5 | conforme tempo |
| Fase 7 (escrita) | E8–E10 | trabalho futuro |

Ver `research-roadmap.md` para prazos, marcos e critérios de paragem.
