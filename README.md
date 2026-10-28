# Base de pesquisa — DRL para energy saving em O-RAN

Trabalho de base para uma dissertação de mestrado sobre o artigo
**“Design and Evaluation of Deep Reinforcement Learning for Energy Saving in
Open RAN”** (Bordin et al., IEEE CCNC 2025, arXiv:2410.14021), com auditoria da
implementação local, revisão de literatura e plano de reprodução/experimentação.

> **Estado (24 Out 2026).** Fases A (entendimento), B (auditoria), F (literatura) e
> a análise quantitativa dos dados reais estão concluídas. As decisões D1–D5 foram
> aprovadas e implementadas em `ns-o-ran-gym` (energia EARTH em watts, custo denso
> com ablação de impulso, normalização por célula, IQL offline e campanha Tier S).
> A campanha Tier S está em curso: 21 baselines reais completos e 19 em falta
> (seeds 558–559 e reposição de 3 abortados). `ns-3-mmwave-oran` não foi alterado
> para D1–D5; `oran-e2sim` não foi tocado.

## Resultado em uma página

- O artigo propõe um xApp de energy saving que liga/desliga 7 células NR
  (`Discrete(128)`), com estado de 85 KPMs e reward escalar (Eq. 2), treinado
  **offline** com PPO e REM-DQN+CQL.
- A implementação local cobre o cenário, o estado, a acção, o reward e o pipeline
  offline. As decisões D1–D4 já estão implementadas: energia EARTH em watts
  (`P_on = 130 + 94·x`, `P_sleep = 75 W`), custo de activação denso com ablação de
  impulso, normalização quantílica por célula e IQL offline para os PPO-1..PPO-5.
  Continuam em aberto o modelo de RF/PHY do `TurnOff`, os 2 KPMs ausentes e a
  campanha completa.
- Nos 21 runs reais de baseline com o modelo D1, a “poupança” de energia acompanha
  a perda de throughput: a intensidade energética **piora** 9,3% (Static), 14,8%
  (Dynamic) e 97,9% (Random) face ao Always-On.
- A decomposição do reward (dataset D1–D3 calibrado) mostra que `bs_on` responde
  por **40%** da massa negativa (empatado com a energia); com os pesos da Tabela I,
  o **Random** é o óptimo do reward em **5 dos 6** agentes — apesar do pior
  throughput e do pior RLF. A excepção é o PPO-1, que prefere Static.
- Conclusão: o pipeline corre e a base D1–D4 está implementada, mas **ainda não é
  uma reprodução completa** do artigo enquanto a campanha Tier S não terminar e os
  KPMs de PHY/RF em falta não forem resolvidos. O plano em `reproduction-plan.md`
  define o que falta, quanto custa e o que significaria “reprodução exacta”,
  “aproximada” e “novo experimento”.

## Entregáveis

| Ficheiro | Conteúdo | Estado |
|---|---|---|
| `article-understanding.md` | Fase A: metadados, hipóteses, arquitectura, estado/acção/reward, resultados, inconsistências I1–I4, ambiguidades R1–R9, 10 questões abertas | ✅ |
| `mathematical-formulation.md` | Reconstrução simbólica da Eq. (1)–(3), análise dimensional, 9 ambiguidades, mapeamento para o código, especificação mínima fiel | ✅ |
| `implementation-audit.md` | Fase B: proveniência, ~40 componentes com estado, 6 divergências materiais, tipologia exacta/aproximada/novo, acções prioritárias; inclui o estado pós-D1–D4 | ✅ |
| `environment-report.md` | Hardware, versões, build ns-3, stack Python, modelo de execução, custo medido, riscos | ✅ |
| `reproduction-plan.md` | Definições de reprodução, decisões D1–D5, desenho factorial, protocolo por fase, tiers S/M/L, critérios de aceitação, cronograma | ✅ |
| `experiment-configs/` | Manifestos de baselines, dataset, treino, avaliação, modelo de custo e template de decisões | ✅ |
| `reproduction-results.csv` | 24 runs (21 completos + 3 abortados) com throughput, energia proxy (pré-D1), RLF, `bs_on`, activação e deltas | ✅ |
| `analysis/` | Scripts e tabelas reprocessadas: baselines D1 com IC, ablação proxy, por tupla, decomposição do reward, sensibilidade aos pesos | ✅ |
| `related-work-matrix.csv` | 27 trabalhos com DOI validado, venue, método, cenário, contribuição, lacuna, relação e relevância | ✅ |
| `literature-search-log.md` | 22 queries, critérios, duplicatas, funil, limites de acesso, reprodutibilidade | ✅ |
| `screening-log.csv` | 79 registos triados, 27 incluídos / 52 excluídos com motivo | ✅ |
| `research-extensions.md` | 10 extensões candidatas (E1–E10) com hipótese, desenho, custo, risco e priorização | ✅ |
| `research-roadmap.md` | Roadmap de 12 meses, marcos G0–G6, orçamento, contingência, estrutura da tese | ✅ |

### Mapeamento para os entregáveis da missão de revisão (`prompt.txt`)

| Pedido em `prompt.txt` | Onde está |
|---|---|
| `work-understanding.md` | `article-understanding.md` |
| `search-strategy.md` | `literature-search-log.md` §2–§3 |
| `screening-log.csv` | `screening-log.csv` (79 registos, decisão e motivo) |
| `literature-matrix.csv` | `related-work-matrix.csv` (27 trabalhos) |
| `evidence-report.md` | `implementation-audit.md` + `mathematical-formulation.md` |
| `research-gaps.md` | `research-extensions.md` |

A missão de revisão e a missão de reprodução foram consolidadas num único conjunto
de ficheiros para evitar duas bases de verdade. As fases de revisão estão completas;
a segunda fase de pesquisa (extensões) fica dependente de aprovação.

## Como reproduzir a análise

```bash
cd /mnt/legitbet/elioth_sbrc

# testes rápidos do projecto (75 testes, ~3 s)
cd ns-o-ran-gym && .venv/bin/python -m pytest -q \
  tests/test_es_env_unit.py tests/test_real_data.py \
  tests/test_drl_shapes.py tests/test_agent_scripts.py && cd ..

# baselines D1 (watts): reprocessa os runs reais do manifesto (~20 s)
ns-o-ran-gym/.venv/bin/python analysis/recompute_baselines.py

# ablação da Eq. (1): grava *_proxy.* sem sobrescrever as tabelas D1
ns-o-ran-gym/.venv/bin/python analysis/recompute_baselines.py \
  --energy-model proxy --output-suffix _proxy

# reward: verifica a Eq. (2) e decompõe as 2.079 transições D1–D3 (<2 s)
ns-o-ran-gym/.venv/bin/python analysis/reward_decomposition.py
```

Nenhum destes comandos lança ns-3 nem altera os sub-repositórios. Para lançar
simulações, ver `reproduction-plan.md` §4 e `AGENTS.md`.

## Estrutura do workspace

```
.
├── article-understanding.md      # Fase A
├── mathematical-formulation.md   # Eq. (1)–(3), R1–R9
├── implementation-audit.md       # Fase B
├── environment-report.md
├── reproduction-plan.md
├── reproduction-results.csv
├── experiment-configs/           # manifestos + decisões
├── analysis/                     # scripts + tabelas derivadas
├── related-work-matrix.csv
├── literature-search-log.md
├── screening-log.csv
├── research-extensions.md
├── research-roadmap.md
├── ns-o-ran-gym/                 # repo independente (main, 0e8ea58)
├── ns-3-mmwave-oran/             # repo independente (master, 8c42089d)
└── oran-e2sim/                   # repo independente (develop, 951b3e9)
```

Os três sub-repositórios são independentes; o root **não** é um repositório git.
Os estados locais não commitados (`ns-o-ran-gym` 30 entradas, `ns-3-mmwave-oran` 17)
incluem as alterações D1–D4 em `ns-o-ran-gym` e alterações pré-existentes no fork
ns-3; foram preservados.

## Decisões aprovadas e implementadas

| ID | Decisão | Estado |
|---|---|---|
| D1 | Modelo de energia | ✅ EARTH em watts (`130 + 94·x` W, sleep 75 W) como default; Eq. (1) literal disponível com `--energy-model proxy` |
| D2 | Semântica de $\delta$ | ✅ custo denso como default; impulso disponível com `--es-cost-mode impulse` |
| D3 | População da normalização | ✅ por célula, agregada pela média, como default; normalização global em `--no-per-cell-normalization` |
| D4 | Algoritmo offline do PPO | ✅ IQL discreto para PPO-1..PPO-5; DQN mantém REM-DQN+CQL |
| D5 | Escala da campanha | 🟡 Tier S em curso (40 baselines; 21 completos + 19 em falta); Tier M fica condicionado ao piloto, Tier L é inviável (~28 dias) |

A aprovação de D1–D5 e do escopo de código foi dada em 2026-10-22. O trabalho
restante é de campanha e de análise, não de decisão: ver `reproduction-plan.md` §2
e §11 e `experiment-configs/decisions.md`.

## Convenções e política de dados

- Documentos em **português (pt-BR)**; nomes de ficheiro em inglês.
- **Sem dados sintéticos** em qualquer resultado. `--synthetic`, `offline.synthetic_dataset()`
  e `dataset_paper_scale_300k.npz` são apenas smoke tests de pipeline.
- Toda a métrica reportada tem run, seed, comando e data; runs abortados são
  reportados, não escondidos.
- Resultados do artigo, inferências próprias e hipóteses não testadas são
  distinguidos explicitamente em cada documento.
- As alterações em `ns-o-ran-gym` são as D1–D4 descritas neste documento;
  `ns-3-mmwave-oran` não foi alterado para D1–D5 e `oran-e2sim` não foi tocado.
  Qualquer alteração futura exige autorização (G0 do roadmap).
