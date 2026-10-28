# Relatório Final — Plano de Ação Técnica

**Economia de energia em O-RAN com DRL** · implementação local do paper *Design and
Evaluation of Deep Reinforcement Learning for Energy Saving in Open RAN*
(arXiv:2410.14021, IEEE CCNC 2025) · workspace `/mnt/legitbet/elioth_sbrc`.

**Data:** 2026-10-28 · **Escopo:** fechamento dos 4 itens do plano (triagem dos
crashes, formulação CMDP, fronteira de Pareto, abate de KPMs).
**Política de dados:** 100% runs reais do ns-3 (`examples/output/real_runs/`); nenhum
flag `--synthetic`, nenhum dado sintético, nenhum atalho de pipeline (contrato do
`AGENTS.md`).

---

## Sumário executivo

1. **Os crashes da campanha foram triados e o remédio principal está aplicado e
   compilando.** O patch de validação de IHL em `Ipv4Header::Deserialize`
   (`ns-3-mmwave-oran/src/internet/model/ipv4-header.cc:444-448`) está no código e o
   `./ns3 build` foi re-executado neste relato com **exit 0**. Dos 40 jobs da campanha
   (8 políticas × 5 seeds), **34 completaram** os 10 s e **6 quebraram** — todos nas
   políticas com células em sleep (§1.4). A cobertura esperada do patch + fixes
   anteriores é: 2 crashes fechados deterministicamente (família IHL), 1 já fechado
   pelo fix IPv6 de 28/out e 3 SIGSEGVs da família TCP/SACK como **risco residual**,
   que exigem re-execução (§1.3).
2. **A formulação CMDP com restrição de QoS está desenhada, auditada por parecer
   adversarial e revisada** (`docs/plano_tecnico/cmdp_design.md`, 617 linhas, seção
   §8 "Rev. 1"). O parecer derrubou o dual 1:1 com η constante (o custo médio do
   dataset, 0,0757, excede δ=0,05 e o dual satura em λ_max=100 ainda dentro do
   orçamento de 1,2 M passos); o caminho do primeiro ciclo passa a ser a **grade fixa
   de λ ∈ {0..10}** (ajuste A3) com λ_max reduzido a 10 (A4). Implementação QoS
   atrás de flag confirmada: suíte de testes executada neste relato, **87 aprovados**
   (§2.5).
3. **A fronteira de Pareto real está construída e conferida**
   (`examples/output/analysis/pareto_energy_throughput.png` + JSONs). Nenhum ponto —
   publicado ou re-simulado — fica acima da diagonal 1:1: cada ponto percentual de
   energia economizada custa entre ~1,6 pp e ~2,9 pp de vazão. A região viável alta
   (ΔE 10–20% com perda de vazão ≤10–15% e RLF controlado) está **vazia** — é o alvo
   do agente IQL/CQL com restrição de QoS (§3.3).
4. **O estado de 85 dimensões cai para 70 sem perda de informação linear.** Os 15
   graus redundantes (7 × `mac_pdu_64qam`, 7 × `prb_percentage`, 1 × `bs_on`) são
   fechados algébrica e numericamente: rank(X₈₅)=70 medido por SVD nos dois datasets
   reais; o compacto-70 tem posto completo. Implementado como
   `STATE_VARIANTS["compact70"]` em `src/nsoran/drl/config.py:90-133`
   (`docs/plano_tecnico/kpm_audit.md`, §4).
5. **Achado científico transversal:** desligar células NR sem restrição desestabiliza
   a própria pilha de protocolos do simulador — o espaço de ações irrestrito não é
   operacionalmente viável, e métricas de estabilidade devem acompanhar energia,
   vazão e RLF na avaliação de políticas de sleep (§1.6).

---

## 1. Frente 1 — Triagem dos crashes da campanha 8×5

### 1.1 Remédio aplicado (verificado neste relato)

Patch em `ns-3-mmwave-oran/src/internet/model/ipv4-header.cc`, conferido por leitura
direta do arquivo:

- **`ipv4-header.cc:444-448`** — a validação de `Deserialize()`, que antes rejeitava
  apenas `version != 4`, agora também rejeita IHL < 5:
  `if ((verIhl >> 4) != 4 || ihl < 5) { NS_LOG_WARN("Trying to decode a non-IPv4 header or an IPv4 header with IHL < 5, refusing to do it."); return 0; }`
  — `return 0` faz `Ipv4L3Protocol` descartar o datagrama-lixo antes que `IpForward`
  o re-serialize. É a mesma política de descarte que o fork já aplicou em PDCP/SACK/
  IPv6.
- **`ipv4-header.cc:385-391`** — endurecimento de defesa em profundidade aplicado no
  `GetSerializedSize()`: `return std::max<uint16_t>(m_headerSize, 20);` (o
  `// return 5 * 4;` do stock ns-3.38 permanece comentado, linha 389).
- Contexto do defeito, conferido no mesmo arquivo: `Serialize()` escreve 20 bytes
  fixos com IHL=5 hardcoded (`ipv4-header.cc:399-421`); o `Deserialize()` faz
  `m_headerSize = ihl*4` lido do fio sem validar (hoje validado em :444).

**Build:** re-executado neste relato — `cd ns-3-mmwave-oran && ./ns3 build` →
`BUILD_EXIT=0` (compilação incremental, sem erros).

### 1.2 Por que as alternativas foram descartadas (justificativa pelo histórico)

- **"Replay-only" está REFUTADO:** a reprodução sob gdb com o build *atual*
  (`_crash_diag_dyn556`, lib `internet` de 28/out 08:11, posterior a todos os fixes)
  morre no **mesmo t=7,639932242 s**, quadro #8 `WriteU8` (`buffer.h:883`) ←
  `Ipv4Header::Serialize` ← `Packet::AddHeader` ← `IpForward` (frente 1 — diagnóstico
  da pasta `static_2_2_0_3_seed556`; não re-executado por mim neste relato, mas o
  patch acima ataca exatamente essa cadeia).
- **"Workaround" (limitar células OFF simultâneas) não discrimina:** quebraram a
  tupla com 1 célula OFF (`4_2_0_1`, seeds 558/559) e a com 3 OFF (`2_2_0_3`, seed
  556), enquanto seeds 555–557 das **mesmas** políticas completaram. Um cap alteraria
  as baselines de [3] sem nenhuma garantia.
- **"Document-only" não fecha 40/40** (§ Próximos passos, item 1).

### 1.3 Cobertura esperada dos 6 crashes quebrados

| Job (manifest.json) | Causa | Cobertura |
|---|---|---|
| `jobs[26] static_2_2_0_3_seed556` (t=7,64 s) | NS_ASSERT de **escrita** `buffer.h:883`: re-serialização de datagrama IPv4 com IHL=0 em `IpForward` → `Ipv4QueueDiscItem::AddHeader` → `Serialize` (20 B) vs `GetSerializedSize`=0 | **Fechado deterministicamente pelo patch §1.1** |
| `jobs[6] dynamic_2_2_0_3_seed556` (t=7,64 s) | Idem (mesmo padrão de células, mesmo t) | **Fechado deterministicamente pelo patch §1.1** |
| `jobs[5] dynamic_2_2_0_3_seed555` (t=1,1 s) | Família IPv6 (cabeçalho de extensão) | **Já coberto pelo fix IPv6 de 28/out** (repro `_diag_dynamic555` passou o ponto determinístico de morte com 0 asserts; a enxurrada de "Unknown IP type" no stderr continua, só o aborto sumiu) |
| `jobs[18] dynamic_4_2_0_1_seed558`, `jobs[38] static_4_2_0_1_seed558`, `jobs[39] static_4_2_0_1_seed559` (t=9,9 s) | SIGSEGV em `TcpTxBuffer::Update` ← `ProcessOptionSack` ← `ReadOptions` ← `ReceivedAck`, **com os fixes SACK/TxBuffer já presentes** na lib que crashou (coredumps PIDs 2449458/2734264/3078054, pilha idêntica) | **Risco residual** — re-executar com o build atual; se recorrer, capturar com `NS3_LAUNCH_PREFIX="gdb -batch -ex run -ex 'bt 60' --args"` (método do `AGENTS.md`) para pinar a linha exata antes de qualquer segunda correção |

Causa raiz da família IHL (diagnóstico da frente 1, com cadeia comprovada por gdb no
core do próprio crash — systemd-coredump PID 2796478, cmdline idêntica ao run
`--RngRun=556 --bsOn=2 --bsIdle=2 --bsSleep=0 --bsOff=3`): o RF-OFF força rajadas de
handover que truncam PDUs em voo (stderr cheio de "Fragment too small in txonBuffer")
e injetam datagramas malformados pela via S1-U/TUN do EPC
(`EpcSgwPgwApplication::SendToTunDevice` → `RouteInput` → `IpForward`); o core mostra
`m_headerSize=0` (IHL=0), `m_protocol=0` e endereços lixo. Detalhe do diagnóstico:
a tentativa anterior (`_crashed_attempt2`, pré-fix PDCP de 21/out) morria com o
assert de **leitura** `buffer.h:1008` em t=5,72 s — bug PDCP já corrigido; os runs
atuais sobrevivem até 7,64 s e morrem no assert de **escrita**.

Os guards de `src/lte/model/lte-enb-rrc.cc` (cellId==0, handover obsoleto,
`EvictUsers`; 78 insertions confirmadas via `git diff` pela frente 1) **permanecem
preservados e não estão implicados** — nenhum stderr contém "dropping stale handover
event" nem "secondary cell".

### 1.4 Tabela política × seed (40 linhas)

Construída por mim neste relato a partir de
`ns-o-ran-gym/examples/output/real_runs/manifest.json` (script: parse dos 40 jobs —
`name`, `status`, `returncode`, `sim_time_covered_s`, `bsstate_rows`,
`duration_s`). **34 completos / 6 quebrados**, todos os quebrados com `returncode 1`
(driver: "ns-3 exited with code -6: the run is invalid").

| # | Política | Seed | Status | simTime coberto (s) | linhas bsState | duração (s) |
|---|---|---|---|---|---|---|
| 0 | always_on | 555 | ✔ completo | 10,0 | 708 | 5 079,8 |
| 1 | always_on | 556 | ✔ completo | 10,0 | 708 | 4 914,1 |
| 2 | always_on | 557 | ✔ completo | 10,0 | 708 | 4 721,4 |
| 3 | always_on | 558 | ✔ completo | 10,0 | 708 | 5 099,9 |
| 4 | always_on | 559 | ✔ completo | 10,0 | 708 | 4 968,1 |
| 5 | dynamic_2_2_0_3 | 555 | ✗ **quebrado** | 1,1 | 85 | 469,7 |
| 6 | dynamic_2_2_0_3 | 556 | ✗ **quebrado** | 7,6 | 540 | 3 477,9 |
| 7 | dynamic_2_2_0_3 | 557 | ✔ completo | 10,0 | 708 | 4 540,6 |
| 8 | dynamic_2_2_0_3 | 558 | ✔ completo | 10,0 | 708 | 4 342,8 |
| 9 | dynamic_2_2_0_3 | 559 | ✔ completo | 10,0 | 708 | 4 463,2 |
| 10 | dynamic_3_2_0_2 | 555 | ✔ completo | 10,0 | 708 | 4 692,4 |
| 11 | dynamic_3_2_0_2 | 556 | ✔ completo | 10,0 | 708 | 4 601,6 |
| 12 | dynamic_3_2_0_2 | 557 | ✔ completo | 10,0 | 708 | 4 464,5 |
| 13 | dynamic_3_2_0_2 | 558 | ✔ completo | 10,0 | 708 | 4 738,0 |
| 14 | dynamic_3_2_0_2 | 559 | ✔ completo | 10,0 | 708 | 4 894,1 |
| 15 | dynamic_4_2_0_1 | 555 | ✔ completo | 10,0 | 708 | 5 080,4 |
| 16 | dynamic_4_2_0_1 | 556 | ✔ completo | 10,0 | 708 | 4 797,4 |
| 17 | dynamic_4_2_0_1 | 557 | ✔ completo | 10,0 | 708 | 4 692,7 |
| 18 | dynamic_4_2_0_1 | 558 | ✗ **quebrado** | 9,9 | 701 | 4 817,2 |
| 19 | dynamic_4_2_0_1 | 559 | ✔ completo | 10,0 | 708 | 4 953,9 |
| 20 | random | 555 | ✔ completo | 10,0 | 708 | 4 336,5 |
| 21 | random | 556 | ✔ completo | 10,0 | 708 | 4 127,3 |
| 22 | random | 557 | ✔ completo | 10,0 | 708 | 4 126,7 |
| 23 | random | 558 | ✔ completo | 10,0 | 708 | 4 074,8 |
| 24 | random | 559 | ✔ completo | 10,0 | 708 | 4 281,6 |
| 25 | static_2_2_0_3 | 555 | ✔ completo | 10,0 | 708 | 4 751,6 |
| 26 | static_2_2_0_3 | 556 | ✗ **quebrado** | 7,6 | 540 | 3 557,3 |
| 27 | static_2_2_0_3 | 557 | ✔ completo | 10,0 | 708 | 4 479,1 |
| 28 | static_2_2_0_3 | 558 | ✔ completo | 10,0 | 708 | 4 731,1 |
| 29 | static_2_2_0_3 | 559 | ✔ completo | 10,0 | 708 | 4 537,1 |
| 30 | static_3_2_0_2 | 555 | ✔ completo | 10,0 | 708 | 4 653,9 |
| 31 | static_3_2_0_2 | 556 | ✔ completo | 10,0 | 708 | 4 655,5 |
| 32 | static_3_2_0_2 | 557 | ✔ completo | 10,0 | 708 | 4 587,5 |
| 33 | static_3_2_0_2 | 558 | ✔ completo | 10,0 | 708 | 4 716,0 |
| 34 | static_3_2_0_2 | 559 | ✔ completo | 10,0 | 708 | 4 895,7 |
| 35 | static_4_2_0_1 | 555 | ✔ completo | 10,0 | 708 | 4 837,9 |
| 36 | static_4_2_0_1 | 556 | ✔ completo | 10,0 | 708 | 4 809,6 |
| 37 | static_4_2_0_1 | 557 | ✔ completo | 10,0 | 708 | 4 718,5 |
| 38 | static_4_2_0_1 | 558 | ✗ **quebrado** | 9,9 | 701 | 4 819,6 |
| 39 | static_4_2_0_1 | 559 | ✗ **quebrado** | 9,9 | 701 | 4 927,6 |

Leitura: as 6 terminações anômalas concentram-se nas políticas com células em sleep —
Static/Dynamic com as tuplas (2,2,3) e (4,2,1) de [3] —, enquanto **Always On e
Random completaram os 10 s em todas as seeds**. O tempo coberto pelos quebrados
(1,1 / 7,6 / 9,9 s) marca onde a pilha morreu; os SIGSEGVs de t=9,9 s caíram a ~1
indicação do fim do episódio.

### 1.5 Risco residual

Os 3 SIGSEGVs da `4_2_0_1` ocorreram **com as correções SACK/TxBuffer já presentes**
na biblioteca que crashou (objetos de 21/out 16:09, relink 19:28:55, anteriores aos
crashes de 24/out) — não é regressão conhecida. O protocolo definido é: re-executar
com o build atual (que agora inclui o patch IHL); se recorrer, capturar sob gdb com
`NS3_LAUNCH_PREFIX="gdb -batch -ex run -ex 'bt 60' --args"` para pinar a linha exata
antes de qualquer segunda correção. Nenhum segundo patch deve ser escrito antes de
haver backtrace novo.

### 1.6 Achado científico — parágrafo para o paper

> A campanha de avaliação expôs um achado de estabilidade que qualifica o espaço de
> ações do problema de economia de energia: desligar células NR sem restrição
> desestabiliza a própria pilha de protocolos. No lote final de 40 runs (8 políticas
> × 5 seeds), todas as 6 terminações anômalas concentraram-se nas políticas com
> células em sleep — Static/Dynamic com as tuplas (2,2,3) e (4,2,1) de [3] —,
> enquanto Always On e Random completaram os 10 s em todas as seeds. A causa,
> confirmada por coredumps do systemd e por reprodução determinística sob gdb, é uma
> cadeia de corrupção de pacotes: o RF-OFF força rajadas de handover/evicção que
> truncam PDUs em voo no forwarding lossless RLC-AM/PDCP (avisos "Fragment too small
> in txonBuffer") e injetam datagramas malformados na pilha IP, onde defeitos
> latentes do ns-3.38 transformam pacotes recuperáveis em falhas fatais do simulador
> — NS_ASSERT de leitura além do fim do buffer (cabeçalho de extensão IPv6; PDCP
> truncado pós-handover), NS_ASSERT de escrita na re-serialização de um datagrama
> IPv4 com IHL<5 no IpForward e SIGSEGV no scoreboard TCP (`TcpTxBuffer::Update` via
> `ProcessOptionSack`), este último ocorrendo mesmo com as correções de SACK já
> aplicadas. Três desses sítios foram endurecidos no fork — o pacote malformado é
> descartado com aviso em vez de abortar o processo — e a validação de IHL do IPv4
> foi concluída neste ciclo (§1.1). Além do relato de engenharia, o achado é
> cientificamente relevante: o espaço de ações irrestrito não é operacionalmente
> viável — políticas que maximizam o retorno desligando células agressivamente
> (PPO-2 alterna o estado das 7 células a cada indicação de 100 ms, com −52,3% de
> vazão) também maximizam a instabilidade da pilha —, de modo que um xApp real
> requer garantias funcionais de segurança (mínimo de células ativas, histerese de
> ativação) que o custo de ativação δ_i da Eq. (2) só parcialmente captura, e
> métricas de estabilidade devem acompanhar energia, vazão e RLF na avaliação de
> políticas de sleep.

Artefatos desta frente: `ns-3-mmwave-oran/src/internet/model/ipv4-header.cc` (patch),
`ns-o-ran-gym/examples/output/real_runs/manifest.json` (tabela §1.4),
`ns-o-ran-gym/examples/output/real_runs/static_2_2_0_3_seed556/` e
`static_4_2_0_1_seed558/` (stderr, bsState, coredumps referenciados),
`manifest_attempt*_crashed.json` (histórico de tentativas).

---

## 2. Frente 2 — Formulação CMDP com restrição de QoS

### 2.1 O que o design entrega

`docs/plano_tecnico/cmdp_design.md` (617 linhas) formula o problema como CMDP com
restrição explícita de QoS (vazão ≥ SLA, RLF limitado) em duas vias: (a) Lagrangiano
primal-dual no treino offline (§4) e (b) porta/barreira de SLA na recompensa (§5),
com integração arquivo por arquivo (§6), riscos declarados (§7) e a base de
calibração sobre números reais dos 21 runs (§2). Nada de sintético: os números vêm
da tabela `grafana` dos `database.db` reais.

### 2.2 Parecer adversarial incorporado (Rev. 1)

A seção **"§8. Revisão e ajustes"** (`cmdp_design.md:470-596`) incorpora o parecer
adversarial com marcadores "Rev. 1" nas seções afetadas (status, §4.1, §5.1, §6.3,
§6.5, §7) e Apêndice A ampliado (`cmdp_design.md:599-617`). Transparência declarada
no documento: o texto do parecer chegou **truncado no meio da Q2** e isso está
registrado; toda alegação medida do parecer foi **re-executada pela frente** sobre os
mesmos dados reais — números idênticos aos do parecer.

### 2.3 Q1 — o dual 1:1 não converge no offline (aceito como defeito estrutural)

Medições re-executadas pela frente (τ = 53.680 bytes; escala = μ_AO = 76.686;
`cmdp_design.md:492-501`):

| Quantidade | Valor medido |
|---|---|
| Custo hinge médio por política | AO 0,0146 · static 0,0314 · dynamic 0,0504 · random 0,3141 |
| Custo médio do dataset misto (21 runs, 2.058 indicações) | **0,0757 > δ = 0,05** |
| Deriva dual por passo | +2,57×10⁻⁴ (λ ← λ + 0,01·(0,0757 − 0,05)) |
| λ após 50 mil passos | ≈ 12,9, monotônico |
| Saturação em λ_max = 100 | ≈ 388 mil passos — **dentro** do orçamento de 1,2 M |
| Penalidade λc em λ = 12,9 (random) | até 9,1 — domina \|r\| ≈ 0,1–0,5 da Eq. (2) |

Defeito estrutural aceito: com amostrador uniforme com reposição sobre dataset fixo
(`src/nsoran/drl/rem_dqn.py:229`), a média de lote do custo é a **constante do
comportamento** — o dual não enxerga a política e não persegue ponto sela.
**Ajustes A1–A5** (`cmdp_design.md:509-538`), que prevalecem sobre a §4.1:

- **A1** — regra de coerência δ × c̄_D pré-treino (obrigatória, registrada no
  `training_summary.json`);
- **A2** — dual em duas escalas de tempo: crítico de custo V_c(s) por TD + estimativa
  de Ĵc(π_k) sob a política corrente em validação (FQE/Monte-Carlo) + Robbins-Monro
  com teto + critérios de congelamento de λ;
- **A3** — **grade fixa de λ ∈ {0, 0,5, 1, 2, 5, 10} como caminho principal do
  primeiro ciclo** (um treino offline por valor; barato perto dos ~90 min de parede
  por avaliação live);
- **A4** — `qos_lambda_max` 100 → **10**, com guarda da razão λc̄/|r̄| ≤ 5;
- **A5** — correção do wrapper PPO-SB3 (§6.3), que tinha o mesmo vício da média
  acumulada.

### 2.4 Q2 — porta (b): esparsidade, não-markovianidade e B̂sON (aceito)

- **Esparsidade medida** (re-executada; fração de indicações com ρ̄^(W) < τ):
  static/dynamic **25–34%** com W=10 (banda do parecer 30–37%; diferença de
  definição de janela documentada) ⇒ a **porta suave** passa a variante **padrão** e
  a dura vira ablação (`cmdp_design.md:542-557`).
- **Não-markovianidade corrigida:** variantes de estado **full+gate (86 dims)** e
  **compact+gate (79 dims)** com a feature da janela anexada, via `STATE_VARIANTS`
  existente (`src/nsoran/drl/config.py:76-87`) e validação de largura do
  `offline.py:130-146`; no live, wrapper lê
  `info["raw_kpms"]["totals"]["throughput"]` (`src/environments/es_env.py:1588-1594`)
  sem mudar o env nem o contrato de 85 dims (`cmdp_design.md:558-568`).
- **Eq. (5.1) SUPERSEDA pela (5.1′)** (`cmdp_design.md:569-584`): ambos os canais de
  w₂ (γ̂ e B̂sON) agora entram **dentro** da porta — fora dela, a política não é mais
  paga por desligar BS violando o SLA.
- Riscos 7–8 adicionados à §7; tabela de impacto consolidado na **§8.3**
  (`cmdp_design.md:586-595`).

### 2.5 Implementação e testes (verificado neste relato)

A maquinaria QoS/variantes está implementada atrás de flag no código
(`src/nsoran/drl/qos.py`, `rem_dqn.py`, `iql.py`, `config.py`). Suíte executada por
mim neste relato, conforme o `AGENTS.md`:

```bash
cd ns-o-ran-gym && .venv/bin/python -m pytest -q \
  tests/test_es_env_unit.py tests/test_real_data.py \
  tests/test_drl_shapes.py tests/test_agent_scripts.py
# → 87 passed in 2.56s
```

A suíte cresceu de 63 para **87 testes** com os testes adicionados pelas frentes 2 e
4 — todos verdes.

---

## 3. Frente 3 — Fronteira de Pareto energia × vazão

### 3.1 Método e reprodutibilidade

- **Artefato:** `ns-o-ran-gym/examples/output/analysis/pareto_energy_throughput.png`
  (206 KB, gerado por `pareto_build.py` na mesma pasta), com os dados em
  `pareto_families_d1.json`, `pareto_perseed_d1.json`,
  `pareto_always_on_perseed_d1.json`, `pareto_points.{csv,json}`.
- **Método (frente 3):** re-agregação das **34 runs completas** do
  `manifest.json` com a ferramenta canônica `examples/compare_runs.py`, modelo **D1**
  (EARTH watts: p0=130 W, Pmax=20 W, Δp=4,7 W, P_sleep=75 W + custo de ativação
  denso — mesmos defaults dos agregados `baselines_d1d3`, cf. `examples/_model_cli.py:22-39`
  e `src/nsoran/drl/real_data.py:74-89`), referência `always_on`. Nenhum ns-3 novo
  foi executado — só agregação.
- **Conferência independente (frente 3):** re-execução de `compare_runs.py` sobre os
  mesmos 34 runs → **0,00 pp de desvio em todos os pontos verificáveis** (JSON em
  `/tmp/conferidor_pareto_d1.json`). Não re-executada por mim neste relato; eu
  verifiquei os JSONs publicados (tabelas abaixo são dos arquivos, extraídas por mim).

### 3.2 Números por família (modelo D1, referência Always On)

Extraídos por mim de `pareto_families_d1.json`:

| Família | ΔEnergia | ΔVazão | RLF máx | bs_on médio | runs | indicações |
|---|---|---|---|---|---|---|
| always_on | 0,00% | 0,00% | 18 | 7,0 | 5 | 495 |
| static_4_2_0_1 | **+6,09%** | **−10,09%** | 18 | 6,0 | 3 | 297 |
| dynamic_4_2_0_1 | **+6,03%** | **−11,90%** | 18 | 6,0 | 4 | 396 |
| static_3_2_0_2 | +12,05% | −20,00% | 20 | 5,0 | 5 | 495 |
| dynamic_3_2_0_2 | +12,15% | −23,87% | 22 | 5,0 | 5 | 495 |
| static_2_2_0_3 | +18,22% | −31,36% | 26 | 4,0 | 4 | 396 |
| dynamic_2_2_0_3 | +18,25% | −34,92% | 27 | 4,0 | 3 | 297 |
| random | +21,39% | −61,66% | **48** | 3,5 | 5 | 495 |

Limitação declarada (minha, deste relato): as famílias com crashes agregam menos
runs/indicações (297–396 vs 495) porque os 6 quebrados ficaram de fora — igualar as
janelas com `--window` ou completar 40/40 remove a assimetria.

Dispersão entre seeds (extraída por mim de `pareto_perseed_d1.json` /
`pareto_always_on_perseed_d1.json`, dp amostral n−1):

| Família | ΔE% (média ± dp) | ΔThr% (média ± dp) |
|---|---|---|
| static/dynamic (todas as tuplas) | **dp ≤ 0,19 pp** | dp entre 9,98 e 18,90 pp |
| random | 21,39 ± 3,51 | −61,66 ± 9,57 |
| always_on (vazão absoluta por seed) | — | **±20,7%** (7,40 ± 1,53 M) |

**Energia é determinística (D1 depende só das horas de sleep); vazão é estocástica.**

### 3.3 Leitura

- **Quem está na região viável (ΔE > 0 com perda de vazão ≤ 20%):** os pontos reais
  Estático (4,2,1) (+6,1%; −10,1%) e Dinâmico (4,2,1) (+6,0%; −11,9%) — as tuplas
  que desligam 1 única célula — e, entre os publicados, DQN (+1,3%; −9,6%) e PPO-1
  (+1,9%; −16,3%). Estático (3,2,2) cai exatamente na borda (−20,0%). RLF no padrão
  destas runs (máx. 18 UEs no caso (4,2,1), igual ao Always On).
- **Quem "economiza" só derrubando tráfego:** **todos** os pontos — publicados e
  re-simulados — ficam **abaixo da diagonal 1:1** (Δvazão = −Δenergia): cada ponto
  percentual de energia economizada custa entre ~1,6 pp (Estático (4,2,1)) e ~2,9 pp
  (Random, PPO-2) de vazão. Extremos de economia ilusória: Random real (+21,4%;
  −61,7%, RLF máx 48, custo de ativação 1,39 — o maior, por ping-pong) e PPO-2
  publicado (+24,2%; −52,3%).
- **Onde um agente IQL/CQL com restrição de QoS deveria pousar:** na parte **alta e
  vazia** da região viável — ΔE entre ~10–20% com perda de vazão ≤10–15% e RLF
  limitado — região hoje desocupada por qualquer política, publicada ou
  re-simulada. O caminho é claro pela estrutura dos dados: ganhar energia é barato e
  estável (desligar células); o difícil é não pagar em vazão — exatamente o que uma
  política fixa (Static/Dynamic desligam conjunto predeterminado) não consegue. Um
  agente condicionado ao estado deveria escolher **qual** célula dormir a cada
  indicação em função do tráfego/mobilidade, mirando o degrau (4,2,1)→(3,2,2): os
  reais dão +6%→+12% de energia com −10%→−20% de vazão; recuperar metade dessa perda
  via adaptação já coloca o agente em (+10–12%; −10%), dominando todas as baselines
  re-simuladas. A restrição de QoS deve entrar como penalidade Lagrangiana explícita
  (limite duro em vazão ≥ 80–90% do Always On e RLF ≤ 2), não só via pesos w1..w4 da
  Eq. (2) — e a avaliação exige **≥ 5 seeds**: com ruído de ±20 pp, veredictos
  single-seed (como os dirs de debug `ppo2_eval*`) não são conclusivos.

### 3.4 Ressalvas (frente 3)

(i) Os ΔE reais usam o modelo **D1 EARTH**; sob o proxy da Eq. (1) os números mudam
(`baselines.csv`: static −14% de energia) e não reproduzem os 0,7%/4,3% publicados;
(ii) PPO-1 publicado tem inconsistência interna (âncora Static +1,9/−16,3 vs âncora
Dynamic +9,1/−24,5; Fig. 4 ≈ +4,7/−22,6); (iii) PPO-3/DQN vêm da leitura vetorial
validada da Fig. 4 (≤ 0,2 pp de erro em 5 âncoras), pois o texto não publica
absolutos.

---

## 4. Frente 4 — Abate de KPMs do estado (85 → 70)

Relatório completo: `ns-o-ran-gym/docs/plano_tecnico/kpm_audit.md` (201 linhas).
Datasets analisados: `examples/output/real_runs/baselines_dataset.npz` e
`baselines_dataset_d1d3.npz` (2.079 transições / 21 episódios cada, 100% reais,
`metadata.source="ns3"`).

### 4.1 Cortes sem perda (evidência código + estatística real)

| # | KPM removido | Índices (85-dim) | Redundância | Evidência código | Evidência estatística (real) |
|---|---|---|---|---|---|
| 1 | `mac_pdu_64qam_c{2..8}` | 9, 21, 33, 45, 57, 69, 81 | cópia exata de `tb_64qam` (ambas recebem a mesma série `TB.TotNbrDlInitial.64Qam`) | `src/environments/es_env.py:842` vs `:844` (mesma série); comentário "proxy" em `:62` | r=1,0 nos 7 pares nos 2 datasets; max\|diff\|=0,0 bit a bit |
| 2 | `prb_percentage_c{2..8}` | 6, 18, 30, 42, 54, 66, 78 | afim exata de `prb_used` (÷139×100); ranges min-max proporcionais ((0,139)/(0,100)) | `es_env.py:821`; ranges `es_env.py:127-128`; `MinMaxNormalizer` `es_env.py:494-495` | r=1,0 nos 7 pares nos 2 datasets; max\|diff\| normalizado = 1,5×10⁻⁸ (float32) |
| 3 | `bs_on` | 84 | soma exata dos 7 `cell_state`; normalizada, obs[84] = Σ cell_state/7 | `es_env.py:874`; ranges (0,7)/(0,1) `es_env.py:220`/`:133` | max\|obs[84]·7 − Σ cell_state\| = 1,8×10⁻⁷ (float32) |

**Fechamento algébrico-numérico:** SVD executado pela frente nos dois .npz reais
(obs+next, 4.158 linhas): **rank(X₈₅) = 70** (tol 1×10⁻⁸), rank(compacto-78) = 70,
rank(compacto-70) = 70 (**posto completo**); condição ≈ 5,3×10²⁰ (base) e
4,9×10²⁰ (d1d3). Os 15 graus perdidos são **exatamente** 7×`mac_pdu` +
7×`prb_percentage` + 1×`bs_on`. A expectativa anterior 85→78 estava correta, porém
**incompleta** — o compacto-78 ainda carregava 8 colinearidades exatas.

### 4.2 O que NÃO cortar (derivadas não-afins)

`ee_ratio` (razão throughput/energia; só colapsa em r≥0,995 nas células sem evento
OFF no d1d3), `rlf_percentage` (denominador `total_ues` variável — r(rlf_count,
rlf_pct) = 0,70–0,91), `tb_64qam_ratio` (denominadores QPSK/16QAM fora do estado) e
`energy` watts (derivada do modelo EARTH, r 0,995–0,9995 vs `cell_state` no d1d3,
mas componente central da recompensa). Cortes adicionais seriam perda informacional
real (`kpm_audit.md:138-145`).

### 4.3 Impacto esperado para IQL/CQL (~3.400–4.000 transições)

N/d sobe de 40,0 → 48,6 (N=3.400) e de 47,1 → 57,1 (N=4.000) — **+21%** de
transições por dimensão sem coletar nada novo; fim das 15 direções nulas
(regressores de valor com backup fitted-Q deixam de trabalhar com mínimos quadrados
mal-posto); a primeira camada densa (512) cai de 43.520 para 35.840 pesos (−17,6%);
o risco para a fidelidade ao paper é nulo — a Eq. (2) usa os totais brutos de
`info["raw_kpms"]` (`es_env.py:921-927`), não o vetor de estado
(`kpm_audit.md:149-173`).

### 4.4 Implementação (verificada neste relato)

Variante registrada em `src/nsoran/drl/config.py:90-133`:
`E5_COMPACT_DROPPED_KPMS = ("mac_pdu_64qam", "prb_percentage")`,
`E5_COMPACT_DROPPED_GLOBAL = ("bs_on",)`, `STATE_DIM_COMPACT70 = 70` e
`STATE_VARIANTS["compact70"]` (a variante `"compact"`/78 fica para compatibilidade;
`config.py:69-87`). Suíte de testes: **87 aprovados** (§2.5).

---

## Próximos passos

1. **Re-executar os 6 jobs quebrados rumo ao 40/40** com os comandos idênticos já
   registrados no `manifest.json` (jobs 5, 6, 18, 26, 38, 39 — §1.3/§1.4), agora com
   o build que inclui o patch IHL. Orçamento: ~90 min de parede por run de 10 s
   ⇒ ~9 h de máquina no total (6 × ~90 min); os quebrados morreram em 1,1–9,9 s, então
   o custo real por run fica entre ~8 min e ~90 min. Protocolo para os 3 SIGSEGVs da
   `4_2_0_1`: se recorrerem, capturar com
   `NS3_LAUNCH_PREFIX="gdb -batch -ex run -ex 'bt 60' --args"` (suportado por
   `NsOranEnv.start_sim`) para pinar a linha exata **antes** de qualquer segundo
   patch (§1.5).
2. **Re-coletar o dataset com o vetor compacto** após o 40/40:
   `examples/collect_run.py` (políticas `keep`/`random`/`random-seconds`) →
   `examples/collect_real_dataset.py --runs-dir ... --state-variant compact70`, com
   `examples/fit_reward_calibration.py` executado antes da conversão (senão as
   componentes da recompensa comprimem em ≈0,005 — gotcha do `AGENTS.md`). Alvo:
   manter a razão N/d ≥ ~50 e o posto completo 70.
3. **Treinar IQL/CQL com a restrição de QoS** seguindo o design revisado: primeiro
   ciclo com **grade fixa de λ ∈ {0, 0,5, 1, 2, 5, 10}** (A3), λ_max = 10 (A4),
   regra de coerência δ × c̄_D pré-treino (A1: c̄_D = 0,0757 > δ = 0,05 — CMDP
   justificado), porta suave como padrão e variante de estado com a janela (B1/B2;
   observação de integração: `compact70` + gate ⇒ 71 dims). λ adaptativo (A2) só se
   a grade indicar necessidade de λ intermediário fino.
4. **Posicionar o agente no Pareto e revalidar a fronteira:** alvo quantitativo —
   ΔE 10–20% com perda de vazão ≤ 10–15% e RLF ≤ 2 (padrão PPO-1/DQN), região hoje
   vazia (§3.3); avaliação com **≥ 5 seeds** (ruído de vazão ±10–19 pp entre seeds;
   o próprio Always On espalha ±20,7%); regenerar
   `examples/output/analysis/pareto_energy_throughput.png` incluindo o ponto do
   agente, com janelas de indicações igualadas entre famílias (`--window`) quando o
   40/40 fechar.

---

## Apêndice — Verificações executadas neste relato (proveniência)

| Verificação | Comando/ato (executado por mim) | Resultado |
|---|---|---|
| Tabela política × seed | parse de `manifest.json` com `.venv/bin/python` (40 jobs) | 34 ok / 6 crashed, §1.4 |
| Patch IHL no código | leitura de `ipv4-header.cc:380-459` | validação `|| ihl < 5` em :444-448; `max(m_headerSize,20)` em :390 |
| Build pós-patch | `cd ns-3-mmwave-oran && ./ns3 build` | `BUILD_EXIT=0` |
| Suíte de testes (4 arquivos do `AGENTS.md`) | `.venv/bin/python -m pytest -q tests/test_es_env_unit.py tests/test_real_data.py tests/test_drl_shapes.py tests/test_agent_scripts.py` | **87 passed** in 2.56 s |
| Design CMDP Rev. 1 | leitura de `cmdp_design.md` (617 linhas; headings + §8 completa) | §8 em `:470-596`, Apêndice A `:599-617` |
| Pareto | leitura de `pareto_families_d1.json`, `pareto_perseed_d1.json`, `pareto_always_on_perseed_d1.json`; existência do PNG (206 KB) | tabelas §3.2 |
| KPMs | leitura de `kpm_audit.md`; grep de `STATE_VARIANTS`/`compact70` em `config.py` | §4; `config.py:90-133` |

Não executados por mim neste relato (reportados pelas frentes, com artefatos no
disco): cadeias gdb/coredump dos crashes (frente 1), medições do design
CMDP §8 (frente 2, re-executada por ela sobre os mesmos dados reais), reprodução do
`compare_runs.py` com 0,00 pp de divergência (frente 3) e SVD de rank (frente 4,
`kpm_audit.md` §2). Nenhum dado sintético foi usado em nenhum ponto.
