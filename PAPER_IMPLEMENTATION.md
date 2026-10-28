# Implementação do artigo — DRL para economia de energia em Open RAN

Este documento é a especificação de integração para a implementação completa do artigo
`Tradução.md` sobre os repositórios:

- `ns-3-mmwave-oran/` — simulador ns-3 + ns-O-RAN.
- `ns-o-ran-gym/` — ambiente Gymnasium e (a partir desta implementação) treinamento DRL.

O objetivo é reproduzir a cadeia do artigo:

1. cenário ns-3 do artigo (7 gNBs, 63 UEs, 850 MHz, 20 MHz, ISD 1700 m, tráfego misto,
   10 s, período E2 de 100 ms);
2. controle externo de liga/desliga das células via arquivo E2;
3. coleta de KPMs e construção de um dataset offline;
4. ambiente Gymnasium com estado de 85 KPMs, ação discreta de 7 bits (128 ações) e
   recompensa configurável;
5. treinamento PPO (Stable Baselines3) e DQN offline (REM-DQN + CQL);
6. avaliação, baselines e gráficos CDF/trade-off.

## Contratos compartilhados

### Células e ações

- Células NR: `[2, 3, 4, 5, 6, 7, 8]` (a célula 1 é o eNB LTE, não controlada).
- Ação: inteiro `0..127`.
- Bit `i` da ação (LSB = célula 2) vale `1` para célula **ligada** e `0` para desligada.
  Exemplo: ação `127` = todas ligadas; ação `0` = todas desligadas.
- O arquivo de controle lido pelo ns-3 é `es_actions_for_ns3.csv` no formato:
  `timestamp_ms,cellId,hoAllowed` com `hoAllowed=1` (ON) / `0` (OFF).
- O estado atual das células também é registrado em `bsState.txt`:
  `Timestamp UNIX Id State` (State 1 = ON, 0 = OFF).

### Vetor de estado (85 dimensões)

O vetor é composto por 12 KPMs por célula (7 células) + 1 KPM global, na ordem:

Para cada célula `c` em `[2,3,4,5,6,7,8]`, 12 valores:
1. `throughput_dl` — `QosFlow.PdcpPduVolumeDL_Filter`
2. `energy` — consumo proxy `TB.TotNbrDl.1 * P_TX` (P_TX configurável, default 1.0)
3. `ee_ratio` — throughput / energy (ou throughput / TB)
4. `rlf_count` — número de UEs com `L3 serving SINR < -5`
5. `rlf_percentage` — percentual de UEs em RLF
6. `prb_used` — `RRU.PrbUsedDl`
7. `prb_percentage` — `RRU.PrbUsedDl / 139 * 100`
8. `tb_64qam` — `TB.TotNbrDlInitial.64Qam`
9. `tb_64qam_ratio` — 64QAM / (QPSK+16QAM+64QAM)
10. `mac_pdu_64qam` — número de PDUs MAC com MCS 64-QAM (proxy: `TB.TotNbrDlInitial.64Qam`)
11. `es_on_cost` — `0.9 ** (0.01 * T_on_ms)` conforme Eq. do artigo
12. `cell_state` — 1 se ligada, 0 se desligada

Global:
85. `bs_on` — número de células ligadas no instante.

O vetor é normalizado para `[0,1]` (min-max por KPI) antes de entrar na rede neural.
Os componentes da recompensa usam normalização por quantis (uniforme ou normal),
conforme a Tabela I do artigo.

### Recompensa

```
r = w1 * throughput_norm
  - w2 * energy_norm
  - w3 * rlf_norm
  - w4 * es_on_cost_norm
  - w2 * bs_on_norm
```

Pesos da Tabela I:

| agente | w1 | w2 | w3 | w4 | normalização |
|---|---:|---:|---:|---:|---|
| PPO-1 | 0.51 | 0.19 | 0.20 | 0.10 | normal |
| DQN   | 0.40 | 0.40 | 0.10 | 0.10 | uniforme |
| PPO-2 | 0.40 | 0.40 | 0.10 | 0.10 | uniforme |
| PPO-3 | 0.20 | 0.40 | 0.20 | 0.20 | uniforme |
| PPO-4 | 0.40 | 0.32 | 0.18 | 0.10 | uniforme |
| PPO-5 | 0.45 | 0.20 | 0.25 | 0.10 | normal |

## Frentes de trabalho

### Frente A — ns-3 / ns-O-RAN

Arquivos principais:
- `ns-3-mmwave-oran/scratch/scenario-three.cc`
- `ns-3-mmwave-oran/src/mmwave/helper/energy-heuristic.{h,cc}`
- `ns-3-mmwave-oran/src/lte/model/lte-enb-rrc.{h,cc}` (se necessário para callback)
- `ns-3-mmwave-oran/src/lte/model/lte-enb-net-device.cc`
- `ns-3-mmwave-oran/README.md`

Entregas:
1. Nova configuração `configuration=3` no `scenario-three`:
   - 850 MHz, 20 MHz, ISD 1700 m, 1 antena Tx/Rx, 7 gNBs (centro + 6 no anel),
     9 UEs por gNB (`ues=9` → 63 UEs).
2. Novo `trafficModel=4` (mistura do artigo):
   - 25% TCP full-buffer 20 Mbps;
   - 25% UDP bursty ~20 Mbps;
   - 25% TCP bursty 750 kbps;
   - 25% TCP bursty 150 kbps.
3. Novo `positionAllocator=2`:
   - UEs uniformemente distribuídos em disco de raio = ISD em torno de cada gNB;
   - `nBsNoUesAlloc` remove `xi` gNBs do posicionamento (0..3), mantendo 63 UEs.
4. Controle externo de energia:
   - O arquivo `es_actions_for_ns3.csv` deve efetivamente ligar/desligar o front-end
     (`MmWaveEnbNetDevice::TurnOn/TurnOff`), não apenas barrar handover.
   - Implementar callback registrado pelo cenário (sem dependência circular lte→mmwave).
   - Garantir que `bsState.txt` reflita o estado real das células.
5. Baselines:
   - `heuristicType=-1`: Always On.
   - `heuristicType=1` e `2`: estática e dinâmica, com tuplas
     `(bsOn, bsIdle, bsSleep, bsOff)` = `(4,2,0,1)`, `(3,2,0,2)`, `(2,2,0,3)`.
   - `heuristicType=3`: random policy do artigo (número uniforme de células desligadas
     a cada 1 s).
6. Smoke test com simulação curta (1 s, 1 UE/gNB) e build.

### Frente B — ambiente Gymnasium

Arquivos principais:
- `ns-o-ran-gym/src/environments/es_env.py`
- `ns-o-ran-gym/src/environments/scenario_configurations/paper_es_use_case.json`
- `ns-o-ran-gym/src/nsoran/ns_env.py` (correções de ciclo de vida)
- `ns-o-ran-gym/src/nsoran/datalake.py` (somente se necessário)
- `ns-o-ran-gym/src/environments/__init__.py`
- `ns-o-ran-gym/tests/`

Entregas:
1. `EnergySavingEnv` com:
   - `action_space = gym.spaces.Discrete(128)`;
   - `observation_space = gym.spaces.Box(0, 1, shape=(85,), dtype=np.float32)`;
   - `reset(seed=...)` define `RngRun` e inicia uma nova simulação;
   - `step` robusto, sem deadlock, com `info` contendo timestamps e KPMs brutas;
   - fechamento correto do subprocesso e semáforos.
2. Extração das 85 KPMs a partir das tabelas `du`, `gnb_cu_cp`/`lte_cu_cp`, `bsState`.
3. Cálculo de RLF, PRB, 64QAM, EE e custo de ativação conforme o artigo.
4. Normalização min-max do estado e normalizador por quantis para a recompensa.
5. Configuração `paper_es_use_case.json`:
   - `configuration=3`, `trafficModel=4`, `ues=9`, `simTime=10`,
     `indicationPeriodicity=0.1`, `heuristicType=-1`, `controlFileName=...`,
     `useSemaphores=1`, `handoverMode=DynamicTtt`, `outageThreshold=-5`.
6. Testes unitários que não dependam de rodar o ns-3:
   - ação inteira → 7 bits;
   - tamanho/ordem do vetor de estado;
   - recompensa com pesos;
   - normalizador.
7. Manter compatibilidade com o exemplo `examples/energy_saving.py`.

### Frente C — DRL (PPO, REM-DQN + CQL)

Arquivos novos (não alterar `es_env.py`):
- `ns-o-ran-gym/src/nsoran/drl/__init__.py`
- `ns-o-ran-gym/src/nsoran/drl/networks.py`
- `ns-o-ran-gym/src/nsoran/drl/rem_dqn.py`
- `ns-o-ran-gym/src/nsoran/drl/offline.py`
- `ns-o-ran-gym/src/nsoran/drl/ppo.py`
- `ns-o-ran-gym/src/nsoran/drl/train.py`
- `ns-o-ran-gym/src/nsoran/drl/evaluate.py`
- `ns-o-ran-gym/examples/train_offline.py`
- `ns-o-ran-gym/examples/evaluate_policy.py`
- `ns-o-ran-gym/examples/_agent_cli.py` (runner compartilhado dos scripts por agente)
- `ns-o-ran-gym/examples/ppo_1.py` … `ns-o-ran-gym/examples/ppo_5.py` (um por agente PPO da Tabela I)
- `ns-o-ran-gym/examples/dqn.py` (REM-DQN + CQL)
- `ns-o-ran-gym/tests/test_agent_scripts.py`
- `ns-o-ran-gym/pyproject.toml` (dependências opcionais `drl`)
- `ns-o-ran-gym/requirements-drl.txt`

Entregas:
1. Dataset offline em `.npz`: `observations`, `actions`, `rewards`, `next_observations`,
   `terminals`, `episode_ids`, `timestamps`.
2. `OfflineReplayEnv` Gymnasium para o PPO treinar sobre o dataset.
3. PPO com Stable Baselines3:
   - `batch_size=256`;
   - `ent_coef` 0.001–0.003;
   - 1.200.000 timesteps configurável;
   - early stopping após 4 avaliações sem melhora, mínimo 10.000 passos.
4. REM-DQN + CQL em PyTorch:
   - CNN 1D + densas `[512, 512, 256]`;
   - ensemble REM com 3 cabeças;
   - loss CQL conservadora configurável;
   - target network e avaliação periódica.
5. Configs dos 6 agentes da Tabela I.
6. Scripts `train_offline.py` e `evaluate_policy.py` com CLI.
7. Testes de formas/tensores que rodem sem GPU e sem ns-3 (quando torch estiver instalado).
8. Scripts por agente com treino + avaliação: `examples/ppo_1.py` … `examples/ppo_5.py`
   e `examples/dqn.py` (config da Tabela I fixa, dataset offline ou sintético para
   smoke tests, avaliação no ns-3 ao vivo ou replay do dataset).

### Frente D — dados, baselines, avaliação e documentação

Arquivos novos:
- `ns-o-ran-gym/scripts/collect_dataset.py`
- `ns-o-ran-gym/scripts/run_baselines.py`
- `ns-o-ran-gym/scripts/run_paper_experiments.sh`
- `ns-o-ran-gym/scripts/make_plots.py`
- `ns-o-ran-gym/docs/PAPER.md`
- `README_PAPER.md` (raiz)

Entregas:
1. Coleta de dataset a partir de execuções ns-3 (4 baselines × 2 alocações × 3 tuplas).
2. Conversão dos CSVs do ns-3 em transições offline.
3. CDFs de throughput, energia, RLF e custo; gráfico de trade-off (throughput vs energia).
4. Scripts reproduzíveis com seeds e limites de paralelismo.
5. Documentação de instalação, build, execução e limitações.

Status desta implementação:
- Conversão execução ns-3 → dataset: `ns-o-ran-gym/src/nsoran/drl/real_data.py`
  (`dataset_from_ns3_run` / `dataset_from_ns3_runs` / `transitions_from_ns3_run`)
  e CLI `ns-o-ran-gym/examples/collect_real_dataset.py`. Reconstrói as observações
  com as mesmas funções do `EnergySavingEnv` (validação: concordância **exata** com
  o log de um run real) e lê as ações de `EsActions.txt`. Cada run = 1 episódio.
- Resultados dos 6 agentes com o run real disponível:
  `ns-o-ran-gym/examples/output/real_runs/` (ver README local).
- A coleta em escala (4 baselines × 2 alocações × 3 tuplas) segue limitada pelo
  tempo de simulação: cada execução real de 10 s leva ~13 min de parede.

## Critérios de aceitação

- `./ns3 build` continua passando em `ns-3-mmwave-oran/`.
- `python -m compileall ns-o-ran-gym/src` passa.
- Testes unitários do Gym/DRL passam (quando dependências instaladas).
- Uma simulação curta do cenário do artigo gera `cu-*.txt`, `du-*.txt`,
  `bsState.txt` e `es_actions_for_ns3.csv`.
- O ambiente consegue executar um episódio curto com ações aleatórias e retornar
  observação de shape `(85,)`.
- O dataset offline pode ser criado a partir de pelo menos uma simulação e carregado
  pelo `OfflineReplayEnv`.
- Os scripts de treino/evalidação têm `--help` funcional e falham com mensagem clara
  se as dependências opcionais não estiverem instaladas.

## Comandos de referência

```bash
# build ns-3
cd ns-3-mmwave-oran
./ns3 build

# simulação curta do cenário do artigo
./ns3 run "scratch/scenario-three --configuration=3 --trafficModel=4 --ues=1 \
  --simTime=1 --indicationPeriodicity=0.1 --heuristicType=-1 \
  --enableE2FileLogging=1"

# instalar o Gym
cd ../ns-o-ran-gym
pip install -e .

# teste do ambiente (quando ns-3 estiver compilado)
python examples/energy_saving.py --ns3_path ../ns-3-mmwave-oran --num_steps 10

# DRL
pip install -r requirements-drl.txt
python examples/train_offline.py --help

# DRL — um script por agente da Tabela I (treino + avaliação)
python examples/ppo_1.py --synthetic 1000 --total-timesteps 256          # smoke
python examples/ppo_1.py --dataset data/offline.npz --evaluate \
  --ns3-path ../ns-3-mmwave-oran                                        # paper
python examples/dqn.py --eval-only --model runs/DQN/best_model.pt \
  --ns3-path ../ns-3-mmwave-oran
```
