# Relatório de ambiente e capacidade de execução

**Objetivo.** Registrar, de forma verificável, o ambiente em que a auditoria e os
experimentos foram/serão executados: hardware, sistema, versões, build do ns-3,
ambiente Python, modelo de execução, custo medido e riscos. Serve para que
qualquer resultado seja atribuível a uma configuração conhecida e para dimensionar
a campanha de reprodução.

**Data da medição.** 22 de outubro de 2026 (medições pontuais; versões fixas).

---

## 1. Hardware

| Item | Valor | Como foi obtido |
|---|---|---|
| CPU | Intel Core i5-13600K (13ª geração) | `/proc/cpuinfo` |
| Núcleos lógicos | 20 | `nproc` |
| Arquitetura | x86_64 | `uname -m` |
| RAM total | 15 GiB | `free -g` |
| RAM disponível (medição) | 9 GiB | `free -g` |
| Swap | 0 | `free -g` |
| Disco | 234 GB, 207 GB livres (7% usado) | `df -h /mnt/legitbet` |
| GPU | nenhuma (PyTorch CPU) | `torch 2.14.1+cpu` |

**Implicação.** Sem GPU, o treino é limitado por CPU. Para PPO com MLP pequena
(isso é o caso: 85→64→64→128), o gargalo **não** é o treino — é a simulação ns-3.
O DQN com encoder CNN também é pequeno o suficiente para CPU.

---

## 2. Sistema operacional e toolchain

| Item | Versão |
|---|---|
| Distribuição | Ubuntu 26.04.1 LTS |
| Kernel | 7.0.0-38-generic |
| GCC | 15.2.0 (Ubuntu 15.2.0-16ubuntu1) |
| CMake | 4.4.4 |
| Python | 3.12.14 (venv do projeto) |
| `pytest` | 9.1.1 |

O ns-3 3.38 é compilado com GCC 15.2. **Atenção:** o ns-3 3.38 é de 2022 e não
foi testado com GCC 15; os avisos de compilação são esperados. O build atual
funciona (binário de 19/10), então não é preciso recompilar para rodar.

---

## 3. Build do ns-3

| Item | Valor |
|---|---|
| Diretório | `/mnt/legitbet/elioth_sbrc/ns-3-mmwave-oran` |
| Perfil | `default` (não `optimized`) |
| Binário do cenário | `build/scratch/ns3.38.rc1-scenario-three-default` |
| Data do binário | 19/10/2026 11:05 |
| Tamanho do build | 1,1 GB |
| Wrapper | `./ns3` (CMake), **não** `waf` |
| `build/optimized/` | **não existe** |

**Consequência direta.** `EnergySavingEnv(..., optimized=True)` falha, porque
procura `build/optimized/`. Toda invocação deve usar `optimized=False` (default).
Isso está correto para o build atual; não "corrigir" criando um diretório vazio.

**Recompilar** (só se o código C++ mudar):

```bash
cd /mnt/legitbet/elioth_sbrc/ns-3-mmwave-oran
./ns3 build
```

O build completo leva vários minutos; incremental é rápido. O perfil `default`
tem otimização parcial e checagens de asserção — mais lento que `optimized`, mas
é o que existe.

**Artefatos de execução.** `./ns3 run` escreve muitos arquivos de trace no
**diretório corrente**. Para manter artefatos limpos, rodar a partir de
`examples/output/<uuid>/` (o que `NsOranEnv` já faz via `cwd`).

---

## 4. Ambiente Python

`python` não está no `PATH`. Usar sempre o venv do projeto.

| Pacote | Versão | Observação |
|---|---|---|
| Python | 3.12.14 | venv em `ns-o-ran-gym/.venv` |
| numpy | 2.5.4 | |
| pandas | 3.0.6 | |
| gymnasium | 1.4.0 | API Gymnasium, não `gym` |
| torch | 2.14.1+cpu | sem CUDA |
| stable-baselines3 | 2.9.0 | PPO |
| matplotlib | 3.11.2 | plots |
| scipy | 1.18.1 | |
| posix_ipc | 1.3.2 | semáforos nomeados |
| **scikit-learn** | **AUSENTE** | **bloqueia** `QuantileTransformer` do SB3 e qualquer uso de `sklearn` |
| pytest | 9.1.1 | |

**Risco do `sklearn` ausente.** O `RewardNormalizer` local não depende de
`sklearn` (implementa quantis com `numpy`), então a suíte rápida passa. Mas
qualquer script que importe `sklearn.preprocessing.QuantileTransformer` falhará.
Instalar com `.venv/bin/pip install scikit-learn` **se** for necessário — não foi
instalado durante a auditoria para não alterar o ambiente sem autorização.

---

## 5. Estado dos testes

```bash
cd /mnt/legitbet/elioth_sbrc/ns-o-ran-gym
.venv/bin/python -m pytest -q \
  tests/test_es_env_unit.py tests/test_real_data.py \
  tests/test_drl_shapes.py tests/test_agent_scripts.py
# 63 passed in ~2.7 s
```

| Arquivo | Estado | Observação |
|---|---|---|
| `test_es_env_unit.py` | ✅ | usa datalakes falsos (`setup_sim=False`); não precisa de ns-3 |
| `test_real_data.py` | ✅ | constrói run dirs falsos |
| `test_drl_shapes.py` | ✅ | valida 85/128 e REM-DQN |
| `test_agent_scripts.py` | ✅ | valida CLIs |
| `test_check_env.py` | ✗ | importa `nsoran.environments.ts_env` (pacote inexistente) e hardcoda `/workspace/...` |
| `test_time_ts.py` | ✗ | mesmo problema |

**Conclusão.** `pytest` completo **falha na coleta** por dois testes legados. A
suíte de 63 testes é a referência. Consertar os dois legados é trabalho separado
(não fazer sem autorização, pois pode mascarar dependências antigas).

---

## 6. Modelo de execução Python ↔ ns-3

O caminho **não** usa sockets nem E2TERM. É semáforo POSIX + arquivo:

```
Python                              ns-3
  |                                  |
  |-- Popen(scenario-three, cwd=run) |
  |                                  |-- escreve KPMs (cu-*/du-*)
  |                                  |-- post /sem_metrics_<uuid>
  |<-- wait /sem_metrics_<uuid> -----|
  |-- lê KPMs, calcula obs/reward    |
  |-- escreve es_actions_for_ns3.csv |
  |-- post /sem_control_<uuid> ----->|
  |                                  |-- lê e trunca o CSV
  |                                  |-- aplica ON/OFF, agenda handover
  |                                  |-- próximo passo (100 ms)
```

| Parâmetro | Default | Por que importa |
|---|---|---|
| `step_timeout` | 300 s | uma indicação custa ~50 s; timeout menor trunca o episódio |
| `reset_timeout` | 300 s | espera pela **primeira** indicação (attachment inicial é lento) |
| `indicationPeriodicity` | 0,1 s | 100 indicações por run de 10 s |
| `simTime` | 10 s | |
| `ues` | 9 | 63 UEs no total |

**Histórico de falha.** Defaults antigos de 30 s/5 s faziam **todo** run falhar
após um passo. Os valores atuais (300 s) são obrigatórios. Não reduzir.

**Semáforos órfãos.** Se o processo Python morrer sem `close()`, os semáforos
`/sem_metrics_<uuid>` e `/sem_control_<uuid>` ficam no kernel. Em Linux, ficam em
`/dev/shm/sem.*`; remover manualmente se um run futuro travar na inicialização.
`EnergySavingEnv.close()` foi corrigido para ser idempotente.

---

## 7. Custo medido de execução

| Operação | Tempo de parede | Fonte |
|---|---|---|
| 1 indicação (100 ms simulados) | ~50 s | observação |
| 1 run completo (10 s, 99–100 indicações) | 4.100–5.080 s (68–85 min) | `real_runs/manifest.json` |
| 21 runs em paralelo (4–8 workers) | ~1 dia | campanha observada |
| 1 build completo do ns-3 | vários minutos | `build.log` |
| Suíte de 63 testes | 2,7 s | medição |
| PPO smoke (1k sintético, 256 steps) | segundos | `--synthetic` (só pipeline) |

**Modelo de custo para a campanha do artigo:**

```
T_wall ≈ (N_runs × 75 min) / workers
```

| N_runs | 4 workers | 8 workers |
|---|---|---|
| 100 | 1,3 dias | 0,7 dia |
| 500 | 6,5 dias | 3,3 dias |
| 3.000 | 39 dias | 19,5 dias |

**Leitura.** A campanha de ~3.000 simulações do artigo é **inviável** neste
hardware em uma janela de mestrado, a menos que se reduza `simTime`, se aumente o
paralelismo (mais RAM é o limite: 1,8 GB/run ⇒ 8 runs ≈ 14,4 GB, no limite dos
15 GiB) ou se aceite um dataset menor com declaração explícita. Ver
`reproduction-plan.md` §5.

**Armazenamento.** 4,9 GB para 21 runs ⇒ **~233 MB/run** com os traces brutos
(`MmWaveSinrTime.txt`, `RxPacketTrace.txt`, etc.). Só `database.db` ocupa
**2,4 MB**; mantendo `database.db` + `bsState.txt` + `EsActions.txt` + log +
manifesto, cada run fica em **~5 MB**. Com retenção selectiva, 554 runs ≈ **3 GB** e
3.000 runs ≈ **15 GB** — o disco deixa de ser o constrangimento; o **tempo de
parede** (78 min/run) passa a ser o único limite. A política de retenção deve ser
executada **depois** de `collect_real_dataset.py` validar a ingestão, nunca antes.

---

## 8. Riscos e armadilhas do ambiente

| Risco | Evidência | Mitigação |
|---|---|---|
| Caminhos absolutos hardcoded | `examples/energy_saving.py:10,14` (`/mnt/legitbet/...`); testes legados (`/workspace/...`) | sempre passar `--ns3_path`, `--config`, `--output_folder` |
| `build/optimized/` inexistente | `ls build/optimized` | manter `optimized=False` |
| `sklearn` ausente | import falha | instalar só se necessário; o normalizador local não usa |
| Semáforos órfãos após crash | `/dev/shm/sem.*` | `close()` idempotente; limpar manualmente |
| Truncamento por timeout | defaults antigos 30/5 s | manter 300 s |
| Crash residual LTE/TCP | 3/24 runs falharam após as correções | investigar antes de campanha grande |
| Escrita de traces no cwd | `./ns3 run` | rodar em `output/<uuid>/` |
| RAM no limite | 15 GiB, 1,8 GB/run | máx. ~6–7 workers simultâneos com folga |
| Disco | 207 GB livres, 233 MB/run | retenção seletiva |
| Dois testes legados quebram `pytest` | `nsoran.environments` inexistente | usar os 4 arquivos da suíte rápida |
| Figuras do artigo em W/Mbps vs. código em contagens | `mathematical-formulation.md` §4 | não comparar diretamente |

---

## 9. Reprodução do ambiente (checklist)

Para reconstruir este ambiente do zero:

```bash
# 1. sistema
#    Ubuntu 26.04, gcc 15.2, cmake 4.4.4, 20 cores, >=16 GiB RAM

# 2. ns-3
cd /mnt/legitbet/elioth_sbrc/ns-3-mmwave-oran
./ns3 configure --build-profile=default --enable-examples --enable-tests
./ns3 build
ls build/scratch/ns3.38.rc1-scenario-three-default

# 3. Python
cd /mnt/legitbet/elioth_sbrc/ns-o-ran-gym
python3.12 -m venv .venv
.venv/bin/pip install -e '.[drl]'
.venv/bin/python -c "import numpy, pandas, gymnasium, torch, stable_baselines3, posix_ipc"

# 4. sanidade
.venv/bin/python -m pytest -q \
  tests/test_es_env_unit.py tests/test_real_data.py \
  tests/test_drl_shapes.py tests/test_agent_scripts.py
```

**Versões mínimas registradas:** Python 3.12.14, numpy 2.5.4, pandas 3.0.6,
gymnasium 1.4.0, torch 2.14.1, SB3 2.9.0, posix_ipc 1.3.2, ns-3.38.rc1.

---

## 10. Limitações declaradas

1. **Medições pontuais.** RAM/disco variam com outros processos; os valores são
   indicativos, não garantias.
2. **Sem GPU.** Todos os tempos de treino são CPU.
3. **Build `default`.** Os tempos de simulação seriam diferentes em `optimized`.
4. **Sem benchmark de ns-3.** O custo de ~50 s/indicação é observacional, não
   instrumentado com profiler; pode variar com o cenário e a carga da máquina.
5. **Rede.** Acesso à internet funciona para metadados (OpenAlex/Crossref), mas
   muitos full texts IEEE são paywalled — ver `literature-search-log.md`.
