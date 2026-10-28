# Auditoria de implementação — artigo vs. código local

**Escopo.** Comparar, componente por componente, o que Bordin et al. (CCNC 2025,
arXiv:2410.14021) descrevem com o que existe em `/mnt/legitbet/elioth_sbrc/`
(`ns-3-mmwave-oran`, `ns-o-ran-gym`, `oran-e2sim`), e decidir o que pode ser
chamado de reprodução.

**Método.** Leitura do PDF e de `Tradução.md`; inspeção de código com `grep`/`sed`;
execução da suíte rápida (75 testes); leitura do `manifest.json` e dos bancos
SQLite dos runs reais; reconstrução numérica da recompensa a partir de
`baselines_dataset.npz`; smoke real de 1 s com o modelo D1–D3. Nenhum arquivo de
`ns-3-mmwave-oran` ou `oran-e2sim` foi modificado durante a auditoria; as
alterações em `ns-o-ran-gym` são as D1–D4 descritas abaixo.

**Legenda de status**

| símbolo | significado |
|---|---|
| ✅ **confirmado** | o código implementa o que o artigo descreve, com evidência direta |
| ◐ **parcial** | implementa a estrutura, mas com parâmetro, unidade ou população diferente |
| ✗ **ausente** | não existe no código |
| ⚠ **divergente** | existe, mas faz algo semanticamente diferente do texto |

---

## 0. Status pós-D1–D4 (24 Out 2026)

Esta secção actualiza a auditoria original sem apagar o registo histórico. As
linhas §3.1, §3.3, §3.4 e §3.5 descrevem o estado **anterior**; o que está
implementado agora é:

| Divergência | Estado | Evidência |
|---|---|---|
| §3.1 energia proxy | ✅ **resolvida por D1** | `EnergySavingEnv(energy_model="watts")` é o default; `P_on = 130 + 94·x` W, `P_sleep = 75 W`; `--energy-model proxy` mantém a ablação |
| §3.2 KPMs duplicados | ❌ **aberta** | `mac_pdu_64qam` continua a duplicar `tb_64qam`; 83 dimensões de informação em 85 |
| §3.3 δ denso | ✅ **resolvida por D2** | `es_cost_mode="dense"` é o default; `impulse` disponível para ablação |
| §3.4 normalização global | ✅ **resolvida por D3** | `per_cell_normalization=True`, `per_cell_aggregation="mean"`; calibração com 4×7 séries por célula |
| §3.5 PPO replay | ✅ **resolvida por D4** | `nsoran/drl/iql.py`; PPO-1..PPO-5 usam IQL por default; `--offline-algo replay` é ablação; smoke real de 256 passos em 2.079 transições passou |
| §3.6 dominância de `bs_on` | 🟡 **parcial** | caiu de 89,7% para 40% da massa negativa; `corr(reward, bs_on) = −0.953` e a dispersão de `bs_on` ainda domina |
| Sleep mode 1 / RF | ❌ **aberta** | `TurnOff` continua a não modelar RF/PHY; qualquer latência de wake-up é nova implementação |
| Campanha de baselines | 🟡 **21/40** | 19 em falta (5 a correr, 14 em fila); ver `reproduction-plan.md` §0 |

**Leitura.** Quatro das seis divergências materiais foram resolvidas no código; a
duplicata de KPM e o modelo de RF continuam abertas. O reward ainda não é uma
métrica de QoS: a calibração corrigiu a média, não a variância.

---

## 1. Proveniência: de quem é o código?

| Item | Evidência | Conclusão |
|---|---|---|
| `ns-o-ran-gym` | remote `git@github.com:wineslab/ns-o-ran-gym.git`, branch `main`, HEAD `0e8ea58`, **30 entradas não commitadas** | repositório **upstream** do grupo, não o artefato dos autores do artigo |
| `ns-3-mmwave-oran` | remote `https://github.com/wineslab/ns-o-ran-ns3-mmwave`, branch `master`, HEAD `8c42089d`, **17 entradas não commitadas** | fork do ns-O-RAN; as features de energy saving são **reimplementação local** |
| `oran-e2sim` | remote `wineslab/ns-o-ran-e2-sim`, branch `develop`, HEAD `951b3e9`, limpo | não usado no caminho Python↔ns-3 |
| Código dos autores | o PDF não cita repositório, DOI de artefato ou Zenodo de código; a única referência de software é o Gymnasium (`zenodo.org/record/8127025`) | **não há artefato oficial disponível** |

**Consequência.** Não existe "o código do artigo" para comparar. A auditoria
compara o **texto** com uma **reimplementação independente** feita no mesmo
laboratório. Isso muda o tipo de reprodutibilidade possível (§6).

**Alterações locais não commitadas são intencionais** (ver `AGENTS.md`): guards de
handover em `lte-enb-rrc.cc`, correções de SACK/PDCP em `tcp-option-sack.cc`,
`tcp-tx-buffer.cc`, `lte-pdcp.cc`, `mc-enb-pdcp.cc`, e as features de ES em
`scenario-three.cc` / `mmwave-enb-net-device.cc`. Não devem ser revertidas.

---

## 2. Mapa de componentes

### 2.1 Cenário e ns-3

| Componente do artigo | Evidência textual | Arquivo-função | Status | Evidência técnica | Ação necessária |
|---|---|---|---|---|---|
| 7 células NR controladas, IDs 2–8 | §V-A, §IV; $N=7$; 63 UEs | `scratch/scenario-three.cc:657` `nMmWaveEnbNodes=7` | ✅ | 7 nós mmWave; LTE é o nó 0 | — |
| Topologia: 1 gNB co-localizado + 6 no anel | texto diz "seven additional gNBs" (8 no total) | `scenario-three.cc:712–726` | ⚠ | 1 posição central duplicada + `nConstellation = N-1 = 6` no anel | documentar I1; o texto está errado, o código fecha $N=7$ |
| ISD 1700 m | §V-A | `scenario-three.cc:625` `isd = 1700` | ✅ | — | — |
| FR1 850 MHz, 20 MHz | §V-A | `scenario-three.cc:623–624` | ✅ | `centerFrequency=850e6`, `bandwidth=20e6` no caso 3 | — |
| 9 UEs por gNB = 63 | §V-A | `scenario-three.cc:661` `nUeNodes = ues * nMmWaveEnbNodes` | ✅ | com `ues=9` ⇒ 63 | — |
| Random walk 2–4 m/s | §V-A | `scenario-three.cc:311–320`, `:733` | ✅ | defaults `minSpeed=2.0`, `maxSpeed=4.0` | — |
| Mix de tráfego 25/25/25/25 | §V-A | `scenario-three.cc:1092–1120` `case 4` | ✅ | TCP full 20 Mbps, UDP 20 Mbps, TCP 750 kb/s, TCP 150 kb/s | — |
| `simTime = 10 s` | §V-A | JSON `paper_es_use_case.json` | ✅ | — | — |
| Indicação E2 a cada 100 ms | §V-A | JSON `indicationPeriodicity: 0.1` | ✅ | 99–100 indicações por run | — |
| Handover DynamicTtt | §V-A / baselines | JSON `handoverMode: DynamicTtt` | ✅ | — | — |
| E2 NR habilitado | implícito (KPMs das 7 células) | JSON `e2nrEnabled: 1`, `e2du/e2cuUp/e2cuCp: 1` | ✅ | sem isso só a célula LTE produz KPM | manter obrigatório |
| Sleep mode 1 (71 µs de wake-up) | §III | `mmwave-enb-net-device.cc:122–127` `TurnOff` | ✗ | `TurnOff` só faz `SetSecondaryCellHandoverAllowedStatus(OFF)`, `EvictUsersFromSecondaryCell()`, `SetCellState(OFF)` | não há modelo de RF/PHY; qualquer afirmação sobre latência de wake-up é **nova implementação** |
| Modelo de consumo de energia no ns-3 | §IV Eq. (1) | busca por `EnergySource`/`DeviceEnergyModel` no arquivo | ✗ | nenhum módulo de energia instalado | a "energia" é derivada em Python (§2.2) |
| Controle por arquivo + semáforo | não descrito no artigo (o artigo usa E2/RIC) | `scenario-three.cc` + `ns_env.py` | ◐ | funcionalmente equivalente, mas **não é E2**; o `oran-e2sim` não está no caminho | declarar como desvio de arquitetura |
| Baselines Static/Dynamic com tuplas (4,2,1),(3,2,2),(2,2,3) | §V-B | `scenario-three.cc:1159–1218`, `energy-heuristic.cc` | ✅ | `heuristicType` 1/2; tuplas via `run_paper_baselines.py` | — |
| Alvo de SINR 13 dB (baselines) | §V-B | `energy-heuristic.cc:77` | ✅ | `convertedSinr > 73` ⇔ 13 dB (mapeamento 3GPP) | — |

### 2.2 Estado e observação (Python)

| Componente do artigo | Evidência textual | Arquivo-função | Status | Evidência técnica | Ação necessária |
|---|---|---|---|---|---|
| Estado 85-d = $12N+1$ | §IV | `es_env.py:52–68`, `:686` | ◐ | 12 nomes por célula, mas 2 são preenchimentos | ver linha seguinte |
| 12 KPMs por célula, nomes | §IV enumera 8 + ρ,γ | `es_env.py:52–65` | ◐ | `tb_64qam_ratio` e `mac_pdu_64qam` são os "extras"; `mac_pdu_64qam` **duplica** `tb_64qam` (`:658–660`) | declarar que 2 KPMs são desconhecidos; não alegar fidelidade |
| Throughput = bytes PDCP | §IV | `cell_kpis_from_dataframe`, `:597` | ✅ | `QosFlow.PdcpPduVolumeDL_Filter` | — |
| Energia $\gamma_i=EC_iP_{tx}$ | Eq. (1) | `:669` `energy = transport_blocks * p_tx` | ⚠ | `p_tx=1.0`; é contagem de TB, não watts | substituir por modelo em W ou declarar proxy |
| Razão throughput/energia | §IV | `:671` | ◐ | razão de proxy por proxy | herda o desvio de energia |
| RLF: UEs com SINR < −5 dB | §IV | `:620–650` | ✅ | mínimo por UE, limiar configurável | — |
| % de UEs em RLF | §IV | `:676–681` | ✅ | usa `total_ues` do frame ou fallback `ues=9` | — |
| PRB usado e % | §IV | `:602`, `:672` | ✅ | `RRU.PrbUsedDl / 139 × 100` | — |
| TB 64-QAM e razão | §IV | `:604–612`, `:673–678` | ◐ | razão 64QAM/(QPSK+16QAM+64QAM) é escolha local | documentar |
| Bytes na camada física | §IV | — | ✗ | não há coluna PHY no datalake; `mac_pdu_64qam` é proxy | declarar ausente |
| Custo $\delta_i=0.9^{0.01T_{D_i}}$ | §IV | `:1221–1246` | ⚠ | custo **denso** enquanto ON, 0 quando OFF; $T_{D_i}$ = ms desde o último ON | decidir L1 vs L2 (ver `mathematical-formulation.md` §4.3) |
| $B_{sON}$ = nº de BSs ativas | §IV | `:684–685` | ✅ | contagem sobre 7 células | — |
| Normalização do estado em [0,1] | §IV | `MinMaxNormalizer`, `es_env.py:396–464` | ◐ | faixas genéricas (`_DEFAULT_KPI_RANGES`), clip | ajustar em dados reais; declarar |
| Estado Markoviano | não discutido | — | ✗ | contadores instantâneos; história em `cell_on_since` | declarar limitação |

### 2.3 Ação e controle

| Componente do artigo | Evidência textual | Arquivo-função | Status | Evidência técnica | Ação necessária |
|---|---|---|---|---|---|
| $\mathcal{A}$ discreto, 128 ações | §IV | `es_env.py:24–49`, `config.py:24–29` | ✅ | `Discrete(128)`, 7 bits | — |
| Bit $j$ ↔ célula $j+2$, ON=1 | contrato local | `es_env.py:215–267` | ✅ | `action_to_cell_bits`, `cell_bits_to_action` | — |
| Arquivo `es_actions_for_ns3.csv`, header `timestamp,cellId,hoAllowed` | contrato local | `action_controller.py` | ✅ | lido por ns-3 | — |
| Estado real registrado em `bsState.txt` | contrato local | `scenario-three.cc:81`, `:1261` | ✅ | `Timestamp UNIX Id State` | — |
| Timeout de passo > 50 s | correção local | `es_env.py` `step_timeout=300` | ✅ | evita truncar antes do 1º KPM | manter |

### 2.4 Recompensa e treinamento

| Componente do artigo | Evidência textual | Arquivo-função | Status | Evidência técnica | Ação necessária |
|---|---|---|---|---|---|
| Eq. (2) com 5 termos | Eq. (2) | `es_env.py:718–733` | ◐ | literal, **mas** normaliza totais da rede antes de somar | decidir população de normalização (R9) |
| Pesos da Tabela I | Tabela I | `config.py:78–84`, `:126–195` | ✅ | PPO-1..5, DQN idênticos ao artigo | — |
| Normalização Normal/Uniform por agente | Tabela I | `config.py`, `es_env.py:484–525` | ✅ | método correto | população errada (acima) |
| Calibração por quantis em dados reais | §IV (implícito) | `examples/fit_reward_calibration.py` | ◐ | existe, mas **não foi usada** no dataset de baselines | rodar antes de qualquer treino |
| PPO = SB3, batch 256 | §IV | `ppo.py:192–207` | ✅ | `batch_size=256`, `n_steps=2048`, `lr=3e-4` | — |
| `ent_coef` 0.001–0.003 | §IV | `ppo.py:182–185` | ✅ | validado no intervalo | — |
| 1.200.000 timesteps | §IV | `ppo.py` `DEFAULT_TOTAL_TIMESTEPS` | ✅ | — | — |
| Episódio de 100 passos | §IV | `_agent_cli.py` | ✅ | compatível com 99–100 indicações | — |
| Early stop: 4 avaliações sem melhora, mínimo 10k | §IV | `ppo.py:48–155` | ✅ | `patience=4`, `min_timesteps=10000` | — |
| Treino **offline** em dataset | §IV | `offline.py:501` `OfflineReplayEnv` | ⚠ | `step()` **ignora a ação do agente** (`:675–686`) | PPO não aprende $Q(s,a)$; ver §3.5 |
| DQN = REM + CQL | §IV | `rem_dqn.py`, `networks.py` | ✅ | ensemble de K heads, penalidade CQL, alvo REM | — |
| Encoder 1-D CNN + [512,512,256] | §IV | `networks.py:71`, `:113–150` | ✅ | `Conv1d` + 3 densas | — |
| Dataset ~3.000 simulações / >300.000 pontos | §IV | `real_runs/` | ✗ | 21 runs completos, 2.079 transições, 33 ações únicas | campanha de coleta é o gargalo (§4) |
| Desconto $\gamma$ | **não declarado** | `ppo.py:194`, `rem_dqn.py:59` | ◐ | 0.99 (default SB3) | declarar como escolha |
| Avaliação com 500 pontos por agente | §V-C | `evaluate.py` | ◐ | 99 indicações por run; "500" provavelmente = 5 runs × 100 | ver I3 |

### 2.5 Datalake

| Componente do artigo | Evidência textual | Arquivo-função | Status | Evidência técnica | Ação necessária |
|---|---|---|---|---|---|
| KPMs vêm do E2 | §IV | `datalake.py` `SQLiteDatabaseAPI` | ◐ | lê arquivos `cu-*-cell-*.txt`/`du-cell-*.txt`, não mensagens E2 | declarar |
| Tabelas por célula | — | `lte_cu_cp`, `gnb_cu_cp`, `lte_cu_up`, `gnb_cu_up`, `du` | ✅ | cellId do nome do arquivo | — |
| Join por `(timestamp, ueImsiComplete)` | — | `read_kpms()` | ✅ | INNER JOIN | runs com UE ausente perdem linhas |
| Estado RF real | — | tabela `bsState` | ✅ | de `bsState.txt` | — |

---

## 3. Divergências que mudam o resultado

As cinco abaixo não são cosméticas: cada uma altera o sinal que o agente otimiza.

### 3.1 Energia não é energia

> **Estado (24 Out 2026): resolvido por D1.** O default actual é
> `EnergySavingEnv(energy_model="watts")`: `P_on = 130 + 94·x` W, `P_sleep = 75 W`.
> O texto abaixo descreve o estado auditado em 22 Out e fica como histórico; a
> ablação `--energy-model proxy` reproduz o comportamento antigo.

O artigo define $\gamma_i=EC_i\cdot P_{tx,i}$ e reporta watts na Fig. 3b. O código
usa

```python
energy = transport_blocks.astype(float) * float(p_tx)   # p_tx = 1.0
```

(`es_env.py:669`), isto é, **número de transport blocks**. Não há potência,
consumo estático, eficiência de PA nem estado de sono. A "economia de energia"
observada é, na prática, "menos tráfego transportado". Isso é visível na razão
energia/throughput dos baselines (§3.6 de `analysis/baseline_summary.md`):
Always-On 0.00979, Static 0.01039, Dynamic 0.01057, Random 0.01171 — as políticas
"economizam" proporcionalmente ao throughput que deixam de carregar.

**Ação.** Ou (a) implementar um modelo de potência em W (ex.: `P = P_0 + Δp·PRB`
com $P_0$ do sleep mode) e recalibrar, ou (b) assumir explicitamente que a métrica
é um proxy e não comparar com a Fig. 3b.

### 3.2 Dois KPMs do estado são duplicatas

Dos 12 nomes, `mac_pdu_64qam` recebe o mesmo `TB.TotNbrDlInitial.64Qam` que
`tb_64qam` (`es_env.py:658–660`). Portanto a observação de 85 dimensões tem, na
verdade, **83 dimensões de informação** (12→11 por célula, menos a duplicata).
O artigo enumera 10 KPMs e afirma 12; a implementação fecha a conta com uma
duplicata. **Não é recuperação dos 2 KPMs ausentes.**

### 3.3 δ é custo denso, não impulso de ativação

> **Estado (24 Out 2026): resolvido por D2.** O default é `es_cost_mode="dense"`,
> com `impulse` disponível como ablação. O texto abaixo mantém a análise da
> ambiguidade, que continua válida como interpretação.

`_update_es_on_cost` (`es_env.py:1221–1246`) devolve
`0.9 ** (0.01 * ms_desde_o_ultimo_ON)` em **todo** passo em que a célula está ON,
e 0 quando OFF. O artigo chama $\delta$ de "cost to activate" e diz que o custo é
maior quando o gap entre mudanças é pequeno — o que sugere um **impulso** no
instante da ativação. As duas leituras produzem recompensas com esparsidade
diferente:

- densa: penalidade em ~100% dos passos ON;
- impulso: penalidade apenas nos passos de transição OFF→ON.

Com 7 células e ações a cada 100 ms, isso muda a escala do termo em até duas
ordens de grandeza.

### 3.4 Normalização em totais, não por célula

> **Estado (24 Out 2026): resolvido por D3.** O default é
> `per_cell_normalization=True`, `per_cell_aggregation="mean"`; a calibração
> `reward_calibration_d1d3.npz` tem 4×7 séries por célula. O texto abaixo
> documenta a ambiguidade original.

`_build_raw_kpms` produz `"totals"` e `compute_reward` normaliza esses totais.
A Eq. (2), lida como soma em $i$ de termos normalizados, exigiria normalizar
**por célula** e depois somar. Como o normalizador é um quantil (não linear),

$$
\sum_i n(x_i)\neq n\!\Big(\sum_i x_i\Big).
$$

Nenhuma das duas leituras é "a do artigo", porque o artigo não diz qual é. Mas a
escolha precisa ser declarada — e a implementação escolheu a segunda sem
documentar.

### 3.5 PPO offline não é offline RL

> **Estado (24 Out 2026): resolvido por D4.** PPO-1..PPO-5 usam IQL discreto
> (`nsoran/drl/iql.py`) por default; `--offline-algo replay` mantém o
> `OfflineReplayEnv` apenas como ablação. O texto abaixo descreve o problema
> original.

`OfflineReplayEnv.step()` (`offline.py:659–705`) valida a ação, mas **retorna a
transição gravada**:

```python
recorded_action = int(self.dataset.actions[dataset_index])
reward = float(self.dataset.rewards[dataset_index])
observation = self.dataset.next_observations[dataset_index]
```

A ação do agente só aparece em `info["action_mismatch"]`. Consequência formal: o
gradiente do PPO usa $\log\pi(a_{\text{gravada}}\mid s)$ e a vantagem da transição
gravada; não há atribuição de crédito condicional à ação do agente. O que o PPO
aprende é, na prática, **behavior cloning ponderado pela recompensa** — ele
reproduz a política de comportamento (Always-On/Random/Static/Dynamic) e não pode
descobrir uma política melhor que a do dataset.

O REM-DQN+CQL, por outro lado, calcula $Q(s,a)$ para **todas** as ações e aplica a
penalidade CQL, então é um learner offline genuíno. **Os dois agentes não são
comparáveis** na implementação atual.

**Ação.** Para PPO offline, usar um algoritmo que condicione na ação (BCQ, CQL,
IQL, TD3+BC) ou implementar importance sampling/behavior cloning explícito. Para
PPO online, é preciso rodar ns-3 no loop — o que custa ~80 min por episódio.

### 3.6 A recompensa efetiva é dominada por `bs_on`

Medido em `baselines_dataset.npz` (2.079 transições, 21 episódios, pesos PPO-2
0.40/0.40/0.10/0.10, sem calibração):

| termo | contribuição média | % da parte negativa |
|---|---|---|
| $+0.40\,n_\rho$ | $+0.0043$ | — |
| $-0.40\,n_\gamma$ | $-0.0044$ | 1.3% |
| $-0.10\,n_\varsigma$ | $-0.0193$ | 5.8% |
| $-0.10\,n_\delta$ | $-0.0109$ | 3.2% |
| $-0.40\,n_{B_{sON}}$ | $-0.3002$ | **89.7%** |

$\operatorname{corr}(R,n_{B_{sON}})=-0.978$ e $\operatorname{corr}(n_\rho,n_\gamma)=+0.894$.
O erro máximo de reconstrução com esses pesos é $3\times10^{-8}$, o que prova que
o dataset foi gerado **sem arquivo de calibração** — as faixas fallback estão
ativas. O metadado do `.npz` não registra isso.

> **Medição D1–D3 (24 Out 2026): parcialmente resolvido.** O dataset
> `baselines_dataset_d1d3.npz` (2.079 transições, 21 episódios, pesos PPO-2) foi
> recalibrado com 33 séries (5 globais + 4×7 por célula) e o modelo de energia em
> watts. Cada componente normalizada tem média ≈ 0,5, pelo que as contribuições
> médias passam a ser fixadas pelos pesos:

| termo | contribuição média | % da parte negativa |
|---|---|---|
| $+0.40\,n_\rho$ | $+0.2000$ | — |
| $-0.40\,n_\gamma$ | $-0.2000$ | 40% |
| $-0.10\,n_\varsigma$ | $-0.0500$ | 10% |
| $-0.10\,n_\delta$ | $-0.0500$ | 10% |
| $-0.40\,n_{B_{sON}}$ | $-0.2000$ | 40% |

O erro máximo de reconstrução é $4.77\times10^{-8}$. A dominância de `bs_on` na
**média** caiu de 89,7% para 40%, mas a sua **dispersão** continua a ser a maior
(dp 0,278 vs. 0,10–0,19 das restantes), pelo que
$\operatorname{corr}(R,n_{B_{sON}})=-0.953$ e
$\operatorname{corr}(R,n_\rho)=-0.527$. O reward ainda está anti-correlacionado
com o throughput; a calibração corrigiu a escala, não a semântica.

---

## 4. Estado da campanha de dados

| Item | Valor | Fonte |
|---|---|---|
| jobs no manifesto | 24 | `real_runs/manifest.json` |
| concluídos | 21 | idem |
| falhos | 3 | idem |
| transições no dataset | 2.079 | `baselines_dataset.npz` |
| episódios | 21 | idem |
| ações únicas | 33 de 128 | idem |
| duração de um run completo | 4.100–5.080 s (~70–85 min) | manifesto |
| RSS por run | ~1,8 GB | observação |
| paralelismo | 4–8 workers (docs divergem) | `paper_runs/README.md` vs. logs |

**Falhas residuais.** `dynamic_2_2_0_3_seed555` aborta em 1,179 s
("read beyond the bounds of the available buffer space"); `dynamic_2_2_0_3_seed556`
e `static_2_2_0_3_seed556` abortam em 7,640 s ("write after end"). As correções de
SACK/PDCP são de 16:08–19:28 de 21/10; os stderr das falhas são de 00:15–02:54 de
22/10 — **as falhas ocorrem depois das correções**. Há um bug de serialização
residual no stack LTE/TCP.

**Deduplicação suspeita.** `dynamic_2_2_0_3_seed556` e `static_2_2_0_3_seed556`
têm fingerprints idênticos (throughput 43.588,632; energia 602,974; RLF 10,711;
`bs_on` 4,0) e o mesmo instante de crash. Isso indica **trajetórias de ação
idênticas** — provavelmente o mesmo `RngRun` foi aplicado antes de a política
divergir. Não usar as duas como amostras independentes.

**Custo da campanha do artigo.** ~3.000 simulações × ~75 min / 8 workers ≈
**19 dias** de wall time (ou ~39 dias com 4 workers), assumindo 100% de
utilização. Isso é o principal limitador de qualquer reprodução em escala.

---

## 5. O que está confirmado, parcial, ausente e divergente

Contagem dos componentes auditados (§2):

| status | nº | exemplos |
|---|---|---|
| ✅ confirmado | 27 | topologia $N=7$, ISD, banda, UEs, tráfego, ação 128, pesos, PPO hyperparams, REM-DQN |
| ◐ parcial | 12 | estado 85 (duplicata), normalização, calibração, datalake, avaliação 500 pontos |
| ✗ ausente | 4 | sleep mode 1 / 71 µs, modelo de energia em ns-3, bytes PHY, dataset de 300k |
| ⚠ divergente | 6 | topologia textual (I1), energia proxy, $\delta$ denso, normalização em totais, PPO offline, controle por arquivo |

**Leitura.** A infraestrutura de cenário e a estrutura do MDP estão **bem
reproduzidas**. A contagem acima é o snapshot de 22 Out; depois de D1–D4, quatro
das seis divergências materiais foram resolvidas no código (energia em W, δ
denso, normalização por célula e PPO com IQL). Continuam abertas a topologia
textual (I1), os 2 KPMs ausentes/duplicados, o modelo de RF/PHY do `TurnOff` e o
controle por arquivo em vez de E2. O que ainda impede uma comparação com a
Tabela VI é a **campanha e o treino**, não a falta de decisões.

---

## 6. Exata vs. aproximada vs. novo experimento

| Critério | Reprodução exata | Reprodução aproximada | Novo experimento |
|---|---|---|---|
| Código dos autores | exigido | não exigido | não exigido |
| Dataset original (3.000 runs) | exigido | não exigido | não exigido |
| Mesmos números das Figs. 3–5 | exigido | **não** (só mesma tendência/ordem) | não se aplica |
| Pesos e dimensões | exigidos | exigidos | livres |
| Unidades de $\rho,\gamma$ | exigidas | **declaradas e justificadas** | livres |
| $\delta$ | exigido | **uma leitura declarada** | livre |
| Normalização | exigida | **população declarada + calibrada** | livre |
| Validade científica | replica | compara com ressalvas | responde pergunta própria |

**Classificação dos artefatos locais:**

1. **Baselines Always-On / Static / Dynamic / Random** (21 runs reais) →
   **novo experimento** com infraestrutura inspirada no artigo. Não são "os
   baselines do artigo": o modelo de energia é outro e o número de seeds é 3, não 5.
2. **PPO-2 e DQN treinados** (`paper_runs/`) → **pipeline smoke test**. O PPO-2
   parou em 50k de 1,2M timesteps com 843 transições; o DQN rodou 20k passos.
   Nenhum resultado deve ser reportado como desempenho de política.
3. **Avaliações live do PPO-2** (`ppo2_eval` vs. `ppo2_eval_fix`) → **inválidas
   para comparação**: 2.375.986 vs. 102.023 de throughput, 1.763 vs. 64 RLF. A
   diferença é grande demais para ser seed; provavelmente uma rodou com estado
   inicial diferente ou com o binário antigo.

**Regra de linguagem para os próximos documentos.** Usar "reprodução" apenas para
o item 1 do parágrafo acima se e somente se a calibração for aplicada e as
unidades forem alinhadas; caso contrário, "reanálise" ou "experimento próprio".

---

## 7. Ações necessárias, por prioridade

| # | Ação | Bloqueia | Custo | Estado (24 Out 2026) |
|---|---|---|---|---|
| 1 | Implementar/declarar modelo de energia em W | qualquer comparação com Fig. 3b | 1–2 dias | ✅ D1 |
| 2 | Rodar `fit_reward_calibration.py` nos 21 runs e reconverter o dataset | treino com sinal útil | horas | ✅ D1–D3 |
| 3 | Declarar a população de normalização (por célula vs. total) e versionar | comparabilidade | horas | ✅ D3 |
| 4 | Resolver o bug residual de serialização (3/24 runs falham) | campanha em escala | 1–3 dias | 🟡 correções no fork; 3 runs a repor na campanha |
| 5 | Substituir PPO offline por algoritmo action-conditional (CQL/IQL/BCQ) | validade do PPO | 1 dia | ✅ D4 (IQL) |
| 6 | Declarar os 2 KPMs ausentes e remover a duplicata | fidelidade do estado | horas | ❌ aberto |
| 7 | Coletar ≥3 seeds por agente da Tabela I | avaliação | ~2 semanas | ❌ não iniciado |
| 8 | Comparar CDFs (não médias) com as Figs. 3–5 | alegação de reprodução | 1 dia | ❌ não iniciado |

Os itens 1–3 e 5 estão concluídos; o item 4 está em curso. Os itens 6–8 continuam
a ser pré-requisitos para qualquer **alegação de reprodução**.

---

## 8. Anexo: comandos de verificação

```bash
# suíte rápida (75 testes, ~2,5 s)
cd /mnt/legitbet/elioth_sbrc/ns-o-ran-gym
.venv/bin/python -m pytest -q \
  tests/test_es_env_unit.py tests/test_real_data.py \
  tests/test_drl_shapes.py tests/test_agent_scripts.py

# proveniência
for d in ns-o-ran-gym ns-3-mmwave-oran oran-e2sim; do
  git -C /mnt/legitbet/elioth_sbrc/$d remote -v | head -1
  git -C /mnt/legitbet/elioth_sbrc/$d rev-parse --short HEAD
  git -C /mnt/legitbet/elioth_sbrc/$d status --porcelain | wc -l
done

# reconstrução da recompensa — dataset antigo (histórico, sem calibração)
cd /mnt/legitbet/elioth_sbrc/ns-o-ran-gym
.venv/bin/python - <<'PY'
import numpy as np
d = np.load('examples/output/real_runs/baselines_dataset.npz', allow_pickle=True)
rc, r = d['reward_components'], d['rewards']
w = [0.40, 0.40, 0.10, 0.10, 0.40]
rec = w[0]*rc[:,0] - w[1]*rc[:,1] - w[2]*rc[:,2] - w[3]*rc[:,3] - w[4]*rc[:,4]
print('max |rec - r| =', np.abs(rec - r).max())   # ~3e-8
print('mean bs_on share =', (w[4]*rc[:,4].mean()) /
      (w[1]*rc[:,1].mean()+w[2]*rc[:,2].mean()+w[3]*rc[:,3].mean()+w[4]*rc[:,4].mean()))
PY

# reconstrução D1–D3 (calibrado; erro ~5e-8)
.venv/bin/python - <<'PY'
import numpy as np
d = np.load('examples/output/real_runs/baselines_dataset_d1d3.npz', allow_pickle=True)
rc, r = d['reward_components'], d['rewards']
w = [0.40, 0.40, 0.10, 0.10, 0.40]
rec = w[0]*rc[:,0] - w[1]*rc[:,1] - w[2]*rc[:,2] - w[3]*rc[:,3] - w[4]*rc[:,4]
print('max |rec - r| =', np.abs(rec - r).max())          # 4.77e-8
print('mean contributions =', [round(w[i]*rc[:,i].mean(), 4) for i in range(5)])
print('corr(reward, bs_on) =', np.corrcoef(r, rc[:,4])[0,1])
print('corr(reward, throughput) =', np.corrcoef(r, rc[:,0])[0,1])
print('corr(throughput, energy) =', np.corrcoef(rc[:,0], rc[:,1])[0,1])
PY
```
