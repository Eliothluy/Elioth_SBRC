# Log de busca e triagem da literatura

**Objetivo.** Documentar, de forma rastreável, como a base bibliográfica deste
projeto foi construída: perguntas, fontes, consultas, critérios, duplicatas,
exclusões, validação de DOI e limites de acesso. O resultado selecionado está em
`related-work-matrix.csv` (27 registros).

**Data da busca.** 22 de outubro de 2026.

**Escopo.** Literatura sobre (i) energy saving em O-RAN/5G, (ii) DRL/offline RL
aplicado a RAN, (iii) infraestrutura de simulação O-RAN/ns-3, (iv) modelos de
potência e sleep modes, (v) algoritmos de offline RL usados ou candidatos.

---

## 1. Perguntas de pesquisa

| ID | Pergunta | Uso no projeto |
|---|---|---|
| Q1 | Como o artigo se posiciona frente ao estado da arte em ES para O-RAN? | originalidade |
| Q2 | Quais trabalhos usam DRL/offline RL para ligar/desligar células? | baselines e comparação |
| Q3 | Quais modelos de potência existem para traduzir estado RF em watts? | corrigir a lacuna R1 |
| Q4 | Quais algoritmos offline são adequados quando o PPO-replay não condiciona na ação? | corrigir §3.5 da auditoria |
| Q5 | Quais plataformas simulam O-RAN com fidelidade de protocolo? | validar a infraestrutura |
| Q6 | Quais extensões recentes (federado, interpretável, multi-xApp) são candidatas? | Fase E |

---

## 2. Fontes e ferramentas

| Fonte | Papel | Cobertura | Limite |
|---|---|---|---|
| **OpenAlex** | descoberta (busca por tópico e por título) | ampla, multi-editora | metadados às vezes incompletos; sem full text |
| **Crossref** | validação de DOI, autores, venue, ano | editoras com DOI registrado | não cobre arXiv (usa DataCite) |
| **DataCite** | validação de DOIs arXiv/Zenodo | preprints e datasets | metadados mínimos (sem abstract) |
| **PDFs locais** | leitura do artigo, tradução e revisão MDPI | 3 documentos | não substitui a leitura dos demais |

**Não foi usada** uma base proprietária (Scopus/WoS/IEEE Xplore search) por falta
de assinatura. Isso é uma limitação de cobertura, não de método: os DOIs foram
validados em registro oficial.

---

## 3. Consultas executadas

### 3.1 Descoberta por tópico (OpenAlex)

Cada consulta foi executada com página de 8 resultados (limite operacional da
coleta), ordenada por relevância. Total: **80 registros**, **71 DOIs únicos**.

| # | Query | Resultados | Úteis para a matriz |
|---|---|---|---|
| 1 | `deep reinforcement learning energy saving open RAN` | 8 | 3 |
| 2 | `O-RAN energy saving xApp cell sleep` | 8 | 4 |
| 3 | `cell sleep strategies energy efficiency base station` | 8 | 2 |
| 4 | `offline reinforcement learning wireless networks` | 8 | 2 |
| 5 | `ns-O-RAN simulating O-RAN 5G systems ns-3` | 8 | 2 |
| 6 | `OpenRAN Gym experimental research platform` | 8 | 3 |
| 7 | `energy efficiency open radio access network survey` | 8 | 3 |
| 8 | `reinforcement learning cell activation deactivation 5G` | 8 | 3 |
| 9 | `conservative Q-learning offline reinforcement learning` | 8 | 1 |
| 10 | `random ensemble mixture offline reinforcement learning` | 8 | 2 |

### 3.2 Busca por título (para preencher lacunas)

12 consultas por título exato, usadas quando o tópico não retornava o trabalho
conhecido (ex.: algoritmos offline, handover mmWave, surveys). Total: **19
registros**, **16 DOIs únicos**.

| Título consultado | Resultado |
|---|---|
| Improved handover through dual connectivity in 5G mmWave mobile networks | Polese et al., 2017 |
| An Optimistic Perspective on Offline Reinforcement Learning | Agarwal et al., 2019 (REM) |
| Conservative Q-Learning for Offline Reinforcement Learning | Kumar et al., 2020 |
| Stable-Baselines3: Reliable Reinforcement Learning Implementations | Raffin et al., 2021 |
| ColO-RAN: Developing ML-Based xApps for Open RAN Closed-Loop Control | Polese et al., 2023 |
| OpenRAN Gym: AI/ML development, data collection, testing for O-RAN on PAWR | Bonati et al., 2023 |
| A Survey on Offline Reinforcement Learning: Taxonomy, Review, Open Problems | Prudencio et al., 2024 |
| Energy Saving for Heterogeneous Wireless Networks: a Mask-Enhanced Offline RL | Ma et al., 2025 |
| Demo: Enabling DRL Research for Energy Saving in Open RAN | Bordin et al., 2025 |
| Scalable ML-Based Approaches for Energy Saving in Densely Deployed Open RAN | Liang et al., 2026 |
| Explainable and Conflict-Aware Cell Sleep Control in O-RAN | Al-Hameed et al., 2026 |
| COMIX: Generalized Conflict Management in O-RAN xApps | Giannopoulos et al., 2025 |

### 3.3 Validação de DOI

- **22 DOIs** validados no Crossref (título, autores, ano, venue, tipo).
- **5 DOIs** validados no DataCite (arXiv: CQL, REM, IQL, TD3+BC, BCQ).
- **2 DOIs** deliberadamente **rejeitados** como falsos positivos:
  - `10.1109/JSAC.2016.2545559` — retornou "Fog Computing May Help to Save Energy
    in Cloud Computing" (tema diferente; a query de ES em RAN casou por palavras).
  - `10.48550/arXiv.2107.05534` — retornou "1st Place Solution for ICDAR 2021
    Competition on Mathematical Formula Detection" (a query de REM casou por
    "mixture", não por RL).

**Regra de validação.** Um registro só entra na matriz se o DOI resolve e o
título retornado pelo registro coincide com o título usado na busca (comparação
normalizada, sem diferenciar maiúsculas/acentos). Título divergente ⇒ exclusão.

---

## 4. Critérios de seleção

### 4.1 Inclusão

Um trabalho entra se satisfizer **todos**:

1. publicado ou preprint com DOI resolvível;
2. escrito em inglês;
3. aborda ao menos um de: ES em O-RAN/5G, sleep mode, DRL/offline RL para RAN,
   modelagem de potência, infraestrutura de simulação O-RAN, handover em mmWave;
4. ano ≥ 2011 (EARTH é o piso histórico do modelo de potência);
5. contribuição primária ou survey de referência (não editorial, não workshop
   report, não demo sem dados).

### 4.2 Exclusão

Contagens exactas do `screening-log.csv` (79 registos triados):

| Motivo | Definição | Nº |
|---|---|---|
| fora do escopo | cloud/fog, C-RAN, IoT, THz, UAV, quantum, scheduling, segurança, diálogo | 11 |
| periférico | sleep/EE/RL relevante, mas sem O-RAN + ES + DRL em conjunto; citável na discussão | 14 |
| variante de método | CQL/IQL/diffusion/sequence-modeling não usados no desenho | 13 |
| duplicata de versão | arXiv vs. periódico, conferência vs. periódico | 6 |
| survey secundário | surveys cobertos pelos 4 surveys incluídos | 3 |
| venue não verificável | revista sem revisão por pares verificável | 3 |
| sem contribuição primária | tese, apêndice de livro | 2 |
| **total excluído** | | **52** |

Dois registos de ruído de query — *Fog Computing May Help to Save Energy*
(`10.1109/JSAC.2016.2545559`) e um artigo de detecção de fórmulas ICDAR
(`10.48550/arXiv.2107.05534`) — foram descartados na pré-triagem, antes do universo
de 74; não constam do CSV.

**Números.** 74 DOIs descobertos + 5 adicionados por busca dirigida = **79
registos** → 27 incluídos na matriz → 52 excluídos com motivo registado.

### 4.3 Duplicatas tratadas explicitamente

| Obra | Versões encontradas | Escolha |
|---|---|---|
| Scalable ML for ES in Densely Deployed O-RAN | arXiv `2604.00201` + IEEE TGCN `10.1109/TGCN.2026.3679951` | versão de periódico |
| Offline RL for Wireless Network Optimization | arXiv `2311.11423` + IEEE conf. 2023 + IEEE TWC 2024 | TWC 2024 |
| OpenRAN Gym | WCNC 2022 `10.1109/WCNC51071.2022.9771908` + Computer Networks `10.1016/j.comnet.2022.109502` | Computer Networks (mais completo) |
| ns-O-RAN | workshop 2023 `10.1145/3592149.3592161` | única versão |
| REM / CQL | arXiv + NeurIPS/PMLR | arXiv (DOI resolvível via DataCite) |

---

## 5. Resultado da triagem

| Etapa | N |
|---|---|
| registros brutos (10 queries × 8) | 80 |
| DOIs únicos após dedup | 71 |
| registros das 12 queries por título | 19 |
| DOIs únicos adicionais | 16 |
| **universo descoberto (OpenAlex)** | **74** |
| adicionados por busca dirigida (referências dos métodos) | 5 |
| **universo total triado** | **79** |
| excluídos (ver `screening-log.csv`) | 52 |
| **incluídos na matriz** | **27** |
| dos quais peer-reviewed | 22 (14 journal + 8 conference) |
| dos quais preprints | 5 (todos arXiv, com DOI DataCite) |
| leitura de full text | 9 |
| somente abstract (paywall) | 18 |

O ficheiro `screening-log.csv` tem **uma linha por registo** (79), com DOI, título,
ano, venue, query de origem, fonte de validação, decisão, motivo de exclusão e nível
de triagem. Os 52 excluídos distribuem-se por: periférico (14), variante de método
(13), fora do escopo (11), duplicata de versão (6), survey secundário (3), venue não
verificável (3), sem contribuição primária (2). A categoria “periférico” significa
que o trabalho é relevante para a discussão, mas não entrou na matriz comparativa
por não partilhar o eixo O-RAN + energy saving + DRL/offline RL.

### 5.1 Composição por relação com o projeto

| relação | nº | significado |
|---|---|---|
| `core-object` | 1 | o próprio artigo |
| `infrastructure` | 4 | simuladores/plataformas usados ou comparáveis |
| `method` | 9 | modelos de potência e algoritmos de offline RL |
| `baseline` | 4 | políticas de sleep/DRL comparáveis |
| `background` | 5 | surveys e tutoriais |
| `extension` | 4 | federado, interpretável, multi-xApp, MARL |

### 5.2 Composição por ano

2011: 1 · 2016: 1 · 2017: 1 · 2018: 1 · 2019: 1 · 2020: 1 · 2021: 2 ·
2022: 2 · 2023: 6 · 2024: 4 · 2025: 4 · 2026: 3.

**Leitura.** 13 dos 27 (48%) são de 2024–2026, o que confirma que o tópico está
ativo e que uma revisão de 2025–2026 precisa cobrir os trabalhos federados,
interpretáveis e de gestão de conflito.

---

## 6. Limites de acesso

| Situação | Nº | Consequência |
|---|---|---|
| PDF completo disponível (arXiv/MDPI/ACM) | 9 | leitura integral |
| Somente abstract (IEEE paywall) | 13 | análise baseada em abstract; sem verificação de detalhes experimentais |
| IEEE open access (abstract + HTML) | 2 | leitura parcial |
| Elsevier paywall | 1 | somente abstract |

**Risco.** Para os 13 trabalhos em paywall, a matriz registra a contribuição
**como declarada no abstract**, não verificada no corpo do artigo. Isso está
marcado na coluna `full_text_access`. Não se deve citar números desses trabalhos
como se tivessem sido auditados.

**Acesso institucional.** Não foi usado VPN/proxy de biblioteca. Se o orientador
tiver acesso IEEE/Elsevier, os 14 paywalls podem ser fechados; a matriz já indica
quais são.

---

## 7. O que a busca revelou sobre o artigo

1. **O artigo é o único que combina** (a) 85 KPMs O-RAN, (b) ação binária por
   célula, (c) treino offline em escala de 300k pontos e (d) avaliação em ns-3
   com protocolo completo. Os concorrentes mais próximos divergem em pelo menos
   um eixo:
   - Dang et al. 2024 (PPO + masking): não é O-RAN/ns-3, ação com offload;
   - Wang et al. 2024 (DQN xApp): estado pobre (RSS), não offline;
   - Liang et al. 2026 (DRL/federado): modelo de energia próprio, não ns-3;
   - Ma et al. 2025 (offline DDQN): rede real, não O-RAN.
2. **A lacuna de modelo de potência é real e antiga.** EARTH (2011) já fornecia o
   mapeamento RF→potência total; o artigo não o usa. Isso não é originalidade
   perdida, é uma oportunidade de correção.
3. **A lacuna de offline RL é metodológica, não bibliográfica.** CQL, IQL, TD3+BC
   e BCQ existem e são citáveis; o problema é que o PPO-replay não é nenhum
   deles. A literatura de offline RL para wireless (Yang et al. 2024) mostra que
   o desempenho depende criticamente da behavior policy — exatamente o que o
   dataset de baselines local ilustra.
4. **Extensões promissoras e ainda abertas:** federado (Liang 2026), safe/
   interpretável (Al-Hameed 2026), conflito multi-xApp (Giannopoulos 2025),
   multi-agente (Razzac 2023). Nenhuma delas resolve a fidelidade do simulador.

---

## 8. Reprodutibilidade da busca

As consultas podem ser repetidas com:

```bash
# descoberta
curl -s "https://api.openalex.org/works?search=deep+reinforcement+learning+energy+saving+open+RAN&per-page=8"

# validação
curl -s "https://api.crossref.org/works/10.1109/CCNC54725.2025.10976108"
curl -s "https://api.datacite.org/dois/10.48550/arXiv.2006.04779"
```

Os resultados brutos da coleta estão em `/tmp/opencode/lit_discovery.json`,
`lit_titles.json`, `lit_crossref.json`, `lit_datacite.json` e `lit_extra.json`
(cache temporário; a matriz é a fonte versionada).

---

## 9. Limitações metodológicas

1. **Não é uma revisão sistemática PRISMA.** Não há dupla triagem, nem protocolo
   registrado, nem cálculo de kappa. É uma revisão **estruturada e rastreável**,
   adequada a uma dissertação, não a um survey.
2. **Cobertura limitada a 8 resultados por query.** Consultas mais amplas podem
   revelar trabalhos não capturados; a saturação foi observada (os mesmos DOIs
   aparecem em queries diferentes), mas não medida formalmente.
3. **Somente inglês.** Trabalhos em português/espanhol sobre ES em O-RAN não
   foram cobertos.
4. **18 de 27 sem full text.** A análise é de abstract para a maioria.
5. **Sem citação reversa completa.** Não foi feita busca "cited by" nem
   "references of references"; é o próximo passo natural se a revisão for
   expandida.
