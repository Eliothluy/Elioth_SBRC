# `analysis/` — análise quantitativa dos dados reais

Esta pasta contém os scripts e as tabelas derivadas das **execuções reais em ns-3**
já existentes em `ns-o-ran-gym/examples/output/real_runs/`. Nada aqui usa dados
sintéticos nem reutiliza tabelas agregadas do projeto: cada número é recalculado a
partir das pastas de execução (`database.db` + `bsState.txt`) ou do dataset offline
`baselines_dataset_d1d3.npz`.

A configuração **principal** é a D1–D3 aprovada em `experiment-configs/decisions.md`:
energia EARTH em watts (D1), custo de activação denso (D2) e normalização
quantílica por célula (D3). As tabelas `*_proxy.*` preservam a ablação da Eq. (1)
literal (`energia = TB.TotNbrDl.1 × p_tx`, sem potência estática), para que a
diferença entre as duas leituras fique auditável.

## Como reproduzir

```bash
cd /mnt/legitbet/elioth_sbrc

# 1. baselines por run, por política e por tupla (reprocessa os runs do manifesto; ~20 s)
ns-o-ran-gym/.venv/bin/python analysis/recompute_baselines.py

# 2. decomposição do reward e sensibilidade aos pesos da Tabela I (<2 s)
ns-o-ran-gym/.venv/bin/python analysis/reward_decomposition.py

# Ablação da Eq. (1): grava baseline_*_proxy.csv/md em vez de sobrescrever as tabelas D1
ns-o-ran-gym/.venv/bin/python analysis/recompute_baselines.py \
  --energy-model proxy --output-suffix _proxy

# Ablação do reward com o dataset antigo (2 079 transições, normalização global e energia proxy)
ns-o-ran-gym/.venv/bin/python analysis/reward_decomposition.py \
  --dataset ns-o-ran-gym/examples/output/real_runs/baselines_dataset.npz \
  --calibration-file ns-o-ran-gym/examples/output/reward_calibration.npz \
  --output-suffix _proxy
```

Ambos os scripts são só de leitura: não escrevem dentro de `ns-o-ran-gym/` nem de
`ns-3-mmwave-oran/`, e recusam-se a inventar runs em falta (marcam `comparable=no`).

## Entradas

| Entrada | Origem | Uso |
|---|---|---|
| `ns-o-ran-gym/examples/output/real_runs/manifest.json` | campanha de baselines (Tier S em curso) | estado, seed, tupla e duração de cada job |
| `.../real_runs/<run>/<uuid>/database.db` + `bsState.txt` | uma execução ns-3 de 10 s | KPIs por indicação |
| `.../real_runs/baselines_dataset_d1d3.npz` | `examples/collect_real_dataset.py` com D1–D3 | 2 079 transições (21 × 99) com as 5 componentes normalizadas do reward |
| `.../reward_calibration_d1d3.npz` | `examples/fit_reward_calibration.py` | quantis de calibração: 5 séries globais + 4×7 por célula |
| `examples/compare_runs.py::run_metrics` | código do projeto | definição de throughput, energia (watts), RLF, activação e `bs_on` |

O `manifest.json` guarda caminhos relativos à raiz de `ns-o-ran-gym/`; o script
resolve-os contra essa raiz e grava caminhos absolutos no CSV. A campanha Tier S
acrescenta 19 runs (seeds 558–559 e reposição de 3 abortados); os números abaixo
são o estágio de 21 runs completos e serão re-agregados no fim.

## Saídas

| Ficheiro | Conteúdo |
|---|---|
| `baseline_per_run.csv` | todos os jobs do manifesto, métricas por run, `comparable`, `note`, caminho absoluto |
| `baseline_summary.csv` | agregado por política: média, desvio-padrão, IC95 (t de Student), Δthr/Δenergia vs Always-On, intensidade E/thr |
| `baseline_by_tuple.csv` | agregado por (política, tupla) — mostra a dispersão que a média esconde |
| `baseline_summary.md` | leitura em Markdown das três tabelas acima, com a lista de runs abortados |
| `baseline_*_proxy.csv` / `.md` | a mesma análise com a energia da Eq. (1) literal (ablação) |
| `reward_decomposition.csv` | por componente: média normalizada, contribuição média, % da massa negativa, correlações (agregada e dentro da política) |
| `reward_decomposition_by_policy.csv` | as mesmas contribuições, separadas por Always-On / Static / Dynamic / Random |
| `reward_weights_sensitivity.csv` | reward médio de cada baseline sob os seis conjuntos de pesos da Tabela I |
| `reward_decomposition.md` | verificação da Eq. (2), decomposição e sensibilidade, em Markdown |
| `reward_*_proxy.csv` / `.md` | a mesma decomposição no dataset antigo (ablação) |

Os hashes SHA-256 são voláteis enquanto a campanha Tier S estiver em curso. A versão
final será fixada depois de converter os 40 runs; para verificar a versão corrente:

```bash
sha256sum analysis/baseline_*.csv analysis/reward_*.csv
```

## Resultados principais

### Baselines (21 runs completos, 99 indicações cada, modelo D1)

| política | runs | throughput | energia (W) | RLF médio | BS ligadas | Δthr | Δenergia | E/thr vs Always-On |
|---|---|---|---|---|---|---|---|---|
| Always-On | 3 | 76 686 | 919.5 | 7.20 | 7.00 | — | — | 1.000 |
| Static | 8 | 62 196 | 814.8 | 12.43 | 5.12 | −18.9% | −11.4% | 1.093 |
| Dynamic | 7 | 59 769 | 823.0 | 11.52 | 5.29 | −22.1% | −10.5% | 1.148 |
| Random | 3 | 31 407 | 745.1 | 16.30 | 3.90 | −59.0% | −19.0% | 1.979 |

- A **intensidade energética** (energia por unidade de throughput) **piora** em todas
  as políticas que desligam células. A poupança absoluta é sobretudo tráfego não
  entregue, não uma rede mais eficiente. Com a potência de sleep de 75 W, desligar
  uma célula poupa no máximo 66,5% (carga máxima) e no mínimo 42,3% (carga zero).
- O Always-On tem variância enorme entre as três seeds (throughput 61.5k–93.3k;
  energia 918.2–921.1 W), pelo que o IC95 da média é ±39 613 / ±3.7 W. Com n=3,
  qualquer diferença inferior a ~20% não é distinguível do ruído de seed.
- As tuplas 4-2-0-1 são as menos agressivas (Δenergia −5.7% a −7.8%) e as 2-2-0-3 as
  mais agressivas (−24% a −31%); a média por política mistura efeitos muito diferentes.
- **Comparação com o artigo:** o artigo reporta +0,7%/+4,3% de poupança para
  Static/Dynamic com −23,5%/−27,6% de throughput. O modelo D1 dá 11,4%/10,5% de
  poupança com −18,9%/−22,1% de throughput; a Eq. (1) literal dá 14,1%/15,9%. Nenhuma
  das duas reproduz os valores do artigo. Isto é uma **ambiguidade de interpretação**
  (o artigo não publica `P_0`, `P_max`, `P_sleep` nem `P_tx`), não um erro de medição;
  ver `mathematical-formulation.md`.

### Reward (2 079 transições reais, normalização por célula)

- A Eq. (2) é reproduzida com erro máximo de `4.77e-08` pelos pesos do **PPO-2**
  (0.40/0.40/0.10/0.10) — o dataset foi convertido com esses pesos, não com os do
  PPO-1 (default do código).
- Depois da normalização quantílica cada componente tem média ≈ 0,5. A contribuição
  **média** é fixada pelos pesos: `bs_on` 40%, energia 40%, RLF 10%, activação 10%.
  A ordenação entre políticas vem da **dispersão**: `bs_on` tem dp 0.278; as restantes
  componentes ficam em 0.10–0.19.
- `corr(reward, bs_on) = −0.953`; `corr(reward, energy) = −0.712`;
  `corr(reward, throughput) = −0.527`. O reward está **anti-correlacionado com o
  throughput** ao nível do dataset.
- Em **5 dos 6** conjuntos de pesos da Tabela I, o **Random** é o óptimo do reward
  médio, apesar de ter o pior throughput e o pior RLF. A excepção é o PPO-1
  (`w2=0.19`), que prefere Static. A Eq. (2), com esta normalização, não codifica a
  restrição de QoS usada na avaliação.

## Flags de qualidade dos dados

| Flag | Efeito | Tratamento |
|---|---|---|
| 3 runs abortados (`dynamic_2_2_0_3_seed555/556`, `static_2_2_0_3_seed556`) | cobrem 1.1 s e 7.6 s em vez de 10 s | excluídos da agregação; listados em `baseline_summary.md`; a campanha Tier S repõe-nos |
| `dynamic_2_2_0_3_seed556` e `static_2_2_0_3_seed556` têm KPIs idênticos | não são amostras independentes; colapso de política determinística sob a mesma seed | não contam como duas seeds |
| Apenas 3 seeds para Always-On/Random no estágio de 21 runs | IC95 largo, sobretudo no Always-On | a campanha Tier S acrescenta 558–559 |
| Janela de 99 indicações | o primeiro report e o *ramp-up* ficam de fora | todos os runs comparados na mesma janela |
| Calibração D1–D3 em `reward_calibration_d1d3.npz` | 5 séries globais + 4×7 por célula | usada na conversão; sem ela as componentes ficam comprimidas |
| Energia D1 = `130 + 94·x` W, sleep 75 W | inclui potência estática; não é a Eq. (1) literal | decisão D1; ablação em `*_proxy` |

## O que estas tabelas **não** mostram

- Não são resultados de nenhum agente DRL: os artefactos treinados existentes
  (`ppo_2_model`, `dqn_model`) usaram um dataset antigo de 843 transições e pararam
  muito antes dos 1.2 M passos do artigo. Servem para validar o pipeline, não para
  comparar com a Tabela VI.
- Não são a Tabela VI do artigo: o artigo reporta 500 pontos de avaliação por
  agente/baseline; aqui há 99 indicações × 3 seeds por baseline no estágio atual e
  nenhuma avaliação de agente.
- A avaliação offline (`--evaluate` sem `--ns3-path`) apenas reproduz transições
  gravadas; não mede o efeito das acções de uma política. Ver
  `implementation-audit.md`, secção de divergências.

## Relação com os outros documentos

- `../reproduction-results.csv` — a mesma matéria por run, mas com os deltas calculados
  contra a média de 3 runs Always-On; esta pasta é a versão reprocessada e com IC.
- `../mathematical-formulation.md` — Eq. (1)–(3), ambiguidades R1–R9 e o significado
  de `bs_on` no reward.
- `../implementation-audit.md` — o que está implementado, o que é interpretação e o
  que falta para a reprodução exata.
- `../reproduction-plan.md` — decisões D1–D5, tiers S/M/L e o protocolo que produziria
  os dados em falta.
- `../experiment-configs/decisions.md` — registo curto das decisões D1–D5.
