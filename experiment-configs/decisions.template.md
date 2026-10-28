# Decisões de especificação (D1–D5)

**Status:** template — preencher e aprovar antes de qualquer alteração de código
ou campanha de coleta.

Preencha uma linha por decisão. A coluna “escolha” deve conter exatamente uma das
opções listadas; a coluna “justificativa” deve citar a evidência
(`article-understanding.md`, `mathematical-formulation.md`,
`implementation-audit.md`).

---

## D1 — Modelo de energia

| Campo | Conteúdo |
|---|---|
| Problema | a Eq. (1) é dimensionalmente aberta; o código usa `TB × 1.0` (proxy), não watts |
| Opções | (a) manter proxy; (b) `P = P_0 + Δp·PRB` em W; (c) EARTH/E3F |
| Escolha | _a preencher_ |
| Justificativa | |
| Impacto esperado | |
| Aprovação (data/autor) | |

**Nota.** Se (b), é preciso definir $P_0$ (potência de sleep), $\Delta p$ (potência
por PRB) e a fonte dos valores. Sem isso, o resultado continua sendo proxy.

---

## D2 — Semântica de $\delta$

| Campo | Conteúdo |
|---|---|
| Problema | “custo de ativar” vs. fórmula decrescente no tempo ligado; Fig. 3d mostra $\delta>1$ |
| Opções | (a) custo denso enquanto ON; (b) impulso na transição OFF→ON |
| Escolha | _a preencher_ |
| Justificativa | |
| Impacto esperado | |
| Aprovação (data/autor) | |

---

## D3 — População da normalização

| Campo | Conteúdo |
|---|---|
| Problema | o artigo não diz se normaliza por célula ou pelo total da rede |
| Opções | (a) totais da rede; (b) por célula e depois soma |
| Escolha | _a preencher_ |
| Justificativa | |
| Impacto esperado | |
| Aprovação (data/autor) | |

**Nota.** (a) é o que o código faz hoje; (b) exige mudança em `_build_raw_kpms`
e em `compute_reward`. A diferença é mensurável e deve entrar como ablation.

---

## D4 — Algoritmo offline do PPO

| Campo | Conteúdo |
|---|---|
| Problema | `OfflineReplayEnv.step` ignora a ação do agente; PPO vira behavior cloning ponderado |
| Opções | (a) manter replay; (b) CQL; (c) IQL; (d) BCQ; (e) TD3+BC |
| Escolha | _a preencher_ |
| Justificativa | |
| Impacto esperado | |
| Aprovação (data/autor) | |

**Nota.** Se (a), nenhum resultado de PPO pode ser chamado de “offline RL”. O DQN
já é action-conditional e não precisa mudar.

---

## D5 — Escala da campanha

| Campo | Conteúdo |
|---|---|
| Problema | 3.000 runs do artigo custam ~28 dias e ~730 GB; o disco tem 207 GB |
| Opções | Tier S (181 runs) / Tier M (554 runs) / Tier L (3.130 runs) |
| Escolha | _a preencher_ |
| Justificativa | |
| Impacto esperado | |
| Aprovação (data/autor) | |

---

## Autorizações

| Item | Autorizado? | Data | Autor |
|---|---|---|---|
| Editar `ns-o-ran-gym/src/**` | | | |
| Editar `ns-3-mmwave-oran/src/**` | | | |
| Criar `scripts/collect_dataset_campaign.py` | | | |
| Rodar campanha de coleta (≥100 runs) | | | |
| Instalar `scikit-learn` no venv | | | |
| Commitar nos sub-repos | | | |

**Nenhuma linha em branco = nenhuma autorização.**
