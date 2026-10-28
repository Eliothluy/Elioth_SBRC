# Entendimento do artigo — *Design and Evaluation of Deep Reinforcement Learning for Energy Saving in Open RAN*

> **Fase A — engenharia reversa do artigo.**
> Documento de trabalho para a base de pesquisa de mestrado. Toda afirmação sobre o
> artigo é ancorada em seção/página do PDF ou no texto extraído; toda inferência é
> marcada como tal. As ambiguidades são classificadas em **erro confirmado**,
> **hipótese de interpretação** ou **informação insuficiente**, conforme o protocolo
> do projeto. Nenhum resultado deste documento foi produzido por simulação nova.

---

## 1. Metadados e proveniência

| Campo | Valor | Verificação |
|---|---|---|
| Título | *Design and Evaluation of Deep Reinforcement Learning for Energy Saving in Open RAN* | PDF, p. 1 |
| Autores | Matteo Bordin, Andrea Lacava, Michele Polese, Sai Satish, Manoj AnanthaSwamy Nittoor, Rajarajan Sivaraj, Francesca Cuomo, Tommaso Melodia | PDF, p. 1 |
| Afiliações | Northeastern University (Institute for the Wireless Internet of Things), Sapienza University of Rome, Mavenir | PDF, p. 1 |
| Veículo | IEEE CCNC 2025 — 22nd Consumer Communications & Networking Conference | Crossref `10.1109/CCNC54725.2025.10976108` |
| DOI | `10.1109/CCNC54725.2025.10976108` | Crossref, validado em 2026-10-22 |
| Preprint | arXiv:2410.14021v1 [cs.NI], 17 out. 2024 | DataCite `10.48550/arXiv.2410.14021` |
| Páginas | 1–6 | Crossref (`page: 1-6`) |
| Tipo | Artigo de conferência (peer-reviewed) | Crossref (`proceedings-article`) |
| Artefatos | ns-O-RAN (simulador) e o *demo* “Enabling DRL Research for Energy Saving in Open RAN” (CCNC 2025). O demo declara que a ferramenta seria aberta “upon acceptance”. O artigo principal **não** fornece link de código, dataset ou configuração. | PDF, §I e §V; abstract do demo `10.1109/CCNC54725.2025.10975928` |

**Observação de proveniência local.** Os três repositórios do workspace
(`ns-3-mmwave-oran`, `ns-o-ran-gym`, `oran-e2sim`) são forks de projetos upstream
(`wineslab/ns-o-ran-ns3-mmwave`, `wineslab/ns-o-ran-gym`, `o-ran-sc/oran-e2sim`).
As funcionalidades de *energy saving* do artigo foram **reimplementadas localmente**;
não são o artefato oficial dos autores. Isso é decisivo para a Fase B: o que se audita
é uma reimplementação, não o código que gerou os números publicados.

---

## 2. Resumo em um parágrafo

O artigo propõe um **xApp de economia de energia para o near-RT RIC** que liga e
desliga os *front-ends* de RF das células NR (modo de sono 1, com tempo de
desativação + reativação de 71 µs), usando **DRL offline**. O estado tem 85 KPMs
(12 por célula × 7 células + 1 global), a ação é um vetor de 7 bits (128 combinações)
e a recompensa combina *throughput*, consumo de energia, RLF e custo de ativação,
com pesos distintos por agente (Tabela I). Os autores estendem o ns-O-RAN para
aplicar o estado das células e coletam ~3.000 simulações (>300.000 pontos) com
quatro heurísticas de linha de base; treinam cinco agentes PPO (Stable-Baselines3) e
um REM-DQN com CQL, e avaliam *online inference* em simulações independentes. O
resultado central é um **trade-off**: o PPO-2 economiza 24,2% de energia ao custo de
52,3% de *throughput*; PPO-1 e DQN dominam as baselines heurísticas em energia e
*throughput*; as baselines economizam pouco (0,7% e 4,3%) porque são conservadoras.

---

## 3. Problema, motivação e hipóteses

### 3.1 Problema declarado

Redes 5G densificadas consomem energia excessiva; o equipamento de RF responde por
~65% do consumo de uma BS. Soluções monolíticas exigem intervenção manual. O O-RAN
permite controle programável via xApps no near-RT RIC e interface E2. **Pergunta
implícita:** é possível um xApp de DRL aprender, a partir de dados históricos, uma
política de ligar/desligar células que economize energia sem degradar excessivamente
o serviço?

### 3.2 Hipóteses (explícitas e implícitas)

| # | Hipótese | Status no artigo |
|---|---|---|
| H1 | DRL aprende uma política de ativação/desativação melhor que heurísticas | Explícita no abstract e §VI; evidência: PPO-1/DQN vs Static/Dynamic |
| H2 | O treinamento offline com dados de simulação é suficiente e mais seguro que online | Explícita em §IV; justificada por [19] (spec O-RAN) |
| H3 | Existe trade-off energia × *throughput* e ele pode ser navegado por pesos de recompensa | Explícita em §VI e Fig. 4 |
| H4 | O custo de ativação δ evita *ping-pong* | Explícita em §IV (“avoid RLFs … and the same cell changing frequently its state”) |
| H5 | Um estado rico de 85 KPMs é necessário/suficiente | Implícita; não há ablação de estado |
| H6 | Os resultados generalizam para além dos seeds de treino | Implícita; avaliação “statistically independent” (§VI), mas sem teste estatístico |

**Lacuna de hipótese:** o artigo não enuncia uma hipótese falsificável sobre *por que*
o DRL deveria superar as heurísticas, nem realiza ablações (estado, recompensa,
normalização, tamanho do dataset). A comparação é apenas entre variantes de pesos.

---

## 4. Arquitetura do sistema (Fig. 1)

```
                         Near-RT RIC
        ┌─────────────────────────────────────────┐
        │  KPM Collector → Preprocessing → ES xApp │
        │                              │  RL model │
        └──────────────┬───────────────┬───────────┘
                       │ E2 (KPM)      │ E2 (control)
             ┌─────────┴───────┐   ┌───┴──────────────┐
             │ LTE eNB (cell 1)│   │ NR gNBs (2..8)   │
             └─────────────────┘   └──────────────────┘
```

- Um **LTE eNB** (âncora NSA) e **N gNBs NR**; um near-RT RIC conectado a cada célula
  LTE e NR via E2.
- O xApp coleta KPMs, pré-processa e gera ações de ativação/desativação.
- A escolha de xApp (near-RT) em vez de rApp (non-RT) é justificada pela necessidade
  de resposta rápida (§III).
- **No código local não há E2AP real**: o control loop é feito por semáforos POSIX e
  arquivos (`es_actions_for_ns3.csv`), e o `oran-e2sim` não é usado. Ver Fase B.

---

## 5. Cenário de simulação (Fig. 2 e §V-A)

| Parâmetro | Valor declarado | Comentário |
|---|---|---|
| Arquitetura | NSA 5G | LTE como âncora |
| Célula LTE | 1 eNB central | cell 1 |
| Células NR | “one gNB co-located with the LTE BS, surrounded by **seven additional** gNBs”, cada um a **1.700 m** do centro | ver inconsistência **I1** |
| Frequência | FR1, **850 MHz**, **20 MHz** de banda | |
| Canal | 3GPP UMi Street Canyon [26] | Dense Urban [25] |
| UEs | **9 por BS**, total **63** | 9 × 7 = 63; compatível com N=7, não com 8 células |
| Posicionamento | uniforme em disco de raio = ISD, centrado no eNB; variante não uniforme exclui ξ ∈ {1,2,3} gNBs | usado no treino; teste só uniforme |
| Mobilidade | *random walk* bidimensional, velocidade uniforme **2–4 m/s** | |
| Tráfego | 25% TCP *full-buffer* 20 Mbps; 25% UDP *bursty* ~20 Mbps; 50% TCP *bursty* (metade 750 kb/s, metade 150 kb/s); fases on/off exponenciais | `trafficModel=4` no código |
| Duração | **10 s** | |
| Periodicidade E2 | **100 ms** | 100 indicações por run |
| Handover | *dynamic TTT*, TTT decresce com a diferença de SINR | [27] |

### Inconsistência I1 — topologia

O texto diz “one gNB co-located … surrounded by seven additional gNBs” (8 células NR),
mas em seguida fixa **N = 7**, **63 UEs = 9 × 7** e ações sobre **7 células** (IDs 2–8),
e a Fig. 2 mostra 6 gNBs no anel + 1 co-localizado. A aritmética só fecha com
**7 células NR no total**: 1 co-localizada + 6 no anel. O código local usa
`nMmWaveEnbNodes = 7` e IDs 2..8. **Classificação: erro confirmado de redação**
(contradição interna), com a interpretação N=7 corroborada por três grandezas
independentes.

---

## 6. Estado (85 KPMs)

### 6.1 O que o artigo declara

- 85 = **12 · N + 1**, N = 7 (§IV).
- O “+1” é `BsON`, o número de BSs ativas.
- São enumerados **10** KPMs por célula (§IV):
  1. razão *throughput*/energia, ρ/γ;
  2. número de UEs em RLF (SINR < −5 dB);
  3. percentual de UEs em RLF;
  4. número de PRBs agendados;
  5. percentual de PRBs agendados;
  6. número de MAC PDUs com 64-QAM;
  7. número de bytes transmitidos na camada física;
  8. custo δ de ativação;
  9. ρ (throughput no PDCP) — definido antes da lista;
  10. γ (consumo de energia) — definido antes da lista.

### 6.2 Inconsistência I2 — dois KPMs não enumerados

10 KPMs enumerados × 7 + 1 = **71**, não 85. Faltam **2 KPMs por célula**. O artigo
não os nomeia. **Classificação: informação insuficiente** (o número 85 é consistente,
a lista não). A implementação local preenche as 12 posições com
`tb_64qam`, `tb_64qam_ratio`, `mac_pdu_64qam` (duplicando o KPM 6) e `cell_state`,
e **omite os bytes de PHY**. Ver Fase B, tabela de auditoria.

### 6.3 Normalização do estado

O artigo afirma que “normal normalization in the range [0, 1] has been applied to each
remaining KPM in the state vector” (§IV). Não informa:
- se a normalização é min-max global, por célula, ou por *quantil*;
- sobre qual conjunto foi ajustada (treino? todos os runs?);
- como valores fora da faixa são tratados (clip?);
- como `BsON` (0..7) vira uma dimensão de `Box(0,1)`.

O código local usa min-max com faixas fixas “generosas” (`throughput 0–1e7`,
`energy 0–1e5`, etc.), o que satura valores atípicos em vez de quebrar o `Box`.
Isso é uma **decisão de engenharia não documentada no artigo**.

---

## 7. Ação

- 7 células controladas → **2⁷ = 128 ações**.
- Ação representada como vetor binário de 7 bits (§IV).
- `a_i(t) ∈ {ON, OFF}` para cada BS i.
- Não há restrição explícita de que ao menos uma célula fique ligada; o artigo não
  discute o caso “todas OFF”. O código local tem guardas no RRC para evitar o crash
  do `EpcX2` nesse caso (ver Fase B).
- O artigo não informa a ordem dos bits nem o mapeamento bit→cell ID. O código usa
  LSB = cell 2 … MSB = cell 8 (`action_to_cell_bits`). **Hipótese de interpretação**
  (a ordem é arbitrária desde que consistente treino/avaliação).

---

## 8. Recompensa (Eq. 2)

Forma literal no PDF:

$$
\max_{a}\ \sum_{t=t_0}^{\infty}\sum_{i=1}^{N}
\left\{
w_1 \rho_i(a_i(t)) - w_2 \gamma_i(a_i(t)) - w_3 \varsigma_i(a_i(t))
- \delta_i(a_i(t))\,w_4
\right\}
- w_2\, B_{sON}(a_i(t))
$$

sujeito a

$$
\sum_{j=1}^{k} W_j = 1, \qquad a_i(t) \in \mathcal{A},\ \forall t.
$$

### 8.1 Componentes

| Símbolo | Significado declarado | Unidade declarada |
|---|---|---|
| ρ_i | bytes transmitidos na camada PDCP pela célula i | bytes (mas Fig. 3a em Mbps) |
| γ_i | EC_i · P_tx,i, com EC = “total number of PDU transmitted” | PDU · W (não é W) |
| ς_i | número de UEs em RLF (SINR < −5 dB) | contagem |
| δ_i | 0,9^(0,01·T_Di), T_Di = tempo em que a célula permanece ativa (ms) | adimensional |
| B_sON | número de BSs ativas no cenário no instante t | contagem (0..7) |

### 8.2 Inconsistências e ambiguidades

| ID | Descrição | Classificação | Impacto |
|---|---|---|---|
| R1 | γ_i = EC_i·P_tx tem dimensão PDU·W, mas a Fig. 3b reporta W (4.000–8.000 W) | **erro confirmado** (dimensional) | o modelo de energia não é reproduzível a partir da Eq. (1); a própria figura não fecha |
| R2 | ρ é “bytes no PDCP”, mas a Fig. 3a reporta Mbps | **erro confirmado** (unidade) | fator de conversão ausente; CDFs não comparáveis diretamente |
| R3 | O termo `− w2·B_sON(a_i(t))` está **fora** do somatório em i, mas depende de i | **erro confirmado** (notação) | não se sabe se é uma vez por passo ou uma vez por célula; o código subtrai uma vez por passo |
| R4 | O mesmo peso w2 é usado para energia e para B_sON | **erro confirmado** (provável duplicação) | dobra efetiva do peso de energia; o código replica isso |
| R5 | δ = 0,9^(0,01·T_D) ∈ (0,1], mas a Fig. 3d mostra δ até 2,5 | **informação insuficiente** | provável soma sobre células ou outra definição; não reproduzível |
| R6 | O texto chama δ de “cost to activate”, mas a fórmula decresce com o tempo **ligado** | **hipótese de interpretação** | o código trata como custo de permanecer ligado; a leitura “impulso na ativação” não é implementada |
| R7 | A soma vai até ∞, sem fator de desconto | **informação insuficiente** | o código usa γ = 0,99 (default do SB3), não declarado |
| R8 | Não há termo explícito para “todas as células OFF” nem restrição de cobertura | **informação insuficiente** | política pode desligar tudo; o código só evita o crash, não o estado |
| R9 | A normalização por quantis é descrita, mas não se diz **o que** é normalizado (por célula? total?) | **informação insuficiente** | o código normaliza totais da rede, tornando Σn(x_i) ≠ n(Σx_i) |

### 8.3 Tabela I — pesos e normalização

| Agente | w1 | w2 | w3 | w4 | Normalização |
|---|---|---|---|---|---|
| PPO-1 | 0,51 | 0,19 | 0,20 | 0,10 | Normal |
| DQN | 0,40 | 0,40 | 0,10 | 0,10 | Uniform |
| PPO-2 | 0,40 | 0,40 | 0,10 | 0,10 | Uniform |
| PPO-3 | 0,20 | 0,40 | 0,20 | 0,20 | Uniform |
| PPO-4 | 0,40 | 0,32 | 0,18 | 0,10 | Uniform |
| PPO-5 | 0,45 | 0,20 | 0,25 | 0,10 | Normal |

Os pesos somam 1 em todas as linhas. O artigo não informa **como** a normalização
“Normal” ou “Uniform” é aplicada (QuantileTransformer do scikit-learn? CDF empírica?
parâmetros?). O código local implementa uma transformação por quantis própria
(`QuantileNormalizer`), com fallback min-max — ver Fase B.

---

## 9. Treinamento

| Item | Declarado | Omissões |
|---|---|---|
| Paradigma | offline (dataset histórico) | — |
| Dataset | ~**3.000 simulações**, **>300.000** pontos, 10 s cada, métricas a cada 100 ms | composição exata por política/tuple; proporção treino/validação/teste; sementes |
| PPO | Stable-Baselines3, batch 256, `ent_coef` 0,001–0,003, máximo 1.200.000 timesteps, episódios de 100 passos, *early stop* após 4 avaliações sem melhora, mínimo 10.000 | `n_steps`, `learning_rate`, `gamma`, `gae_lambda`, `clip_range`, arquitetura da rede, sementes, frequência de avaliação |
| DQN | REM-DQN + CQL; encoder 1D CNN + densas [512, 512, 256] | learning rate, batch, tamanho do ensemble, α do CQL, target update, ε-greedy, número de passos |
| Ambiente | Gymnasium | versão, wrappers, espaço de observação exato |
| Avaliação | “online inference” em simulações independentes; **500 data points por agente/baseline** | ver I3 |

### Inconsistência I3 — o que é “500 data points”?

O texto diz: “Experiments are repeated with different seeds for a total of **500 data
points** for each agent or baseline” e, logo antes, “The CDFs considers the values for
KPMs in **each E2 report**”. A leitura literal de 500 *runs* de 10 s por agente
implicaria 500 × ~70 min ≈ 24 dias **por agente** só de avaliação, o que é implausível
para o porte do trabalho. A leitura consistente é **500 relatórios E2** por agente,
ou seja, **5 runs × 100 indicações** (ou 5 seeds × 10 s). **Classificação: hipótese de
interpretação forte**, não confirmada pelo texto. Consequência: as CDFs têm 500
amostras correlacionadas em 5 blocos, não 500 réplicas independentes; os intervalos de
confiança seriam bem mais largos do que sugere a leitura ingênua.

### Inconsistência I4 — “three” vs “four” baselines

§II diz “we have implemented … **three** different baselines based on [3]”, enquanto
§V-B lista **quatro** estratégias (Random, Static, Dynamic, Always On). Always On não
vem de [3]. **Classificação: erro confirmado de redação** (menor).

---

## 10. Baselines

| Baseline | Definição no artigo | Parâmetros |
|---|---|---|
| Always On | todas as BSs ativas durante toda a simulação | — |
| Random | desativa um número uniformemente aleatório de BSs **a cada 1 s** | distribuição uniforme |
| Static [3] | ordena BSs por UEs conectados com SINR > alvo; ativa as N_On,P1 primeiras; entre as restantes, ordena pela **distância** ao UE mais próximo; ativa N_On,P2; desativa N_Off | (N_On,P1, N_On,P2, N_Off) ∈ {(4,2,1), (3,2,2), (2,2,3)}; SINR alvo 13 dB; R = 2.000 m |
| Dynamic [3] | idem, mas ordena pelo **tempo** que o UE levaria para alcançar a BS (inclui velocidade) | idem |

O artigo não informa como as baselines escolhem **quais** células desligar quando há
empate, nem a semente do Random, nem se as decisões são reaplicadas a cada 100 ms
(Static/Dynamic) — o texto diz “every E2 indication periodicity (i.e., 100 ms)”, mas
o Random é “every second”. O código local implementa exatamente essa distinção.

---

## 11. Resultados reportados

Tabela reconstruída de §VI (texto e Figs. 3–4). Todos os percentuais são relativos ao
Always On, salvo indicação.

| Configuração | Energia vs Always On | Throughput vs Always On | RLF máx. | Observação |
|---|---|---|---|---|
| Always On | 0% (referência) | 0% (referência) | 0 | máximo de throughput/energia/cobertura |
| Static | **−0,7%** (economia) | **−23,5%** | ≤ 3 UEs | conservadora; baixo custo de ativação |
| Dynamic | **−4,3%** (economia) | **−27,6%** | ≤ 3 UEs | idem |
| PPO-2 | **−24,2%** | **−52,3%** | não reportado no texto | maior economia; maior número de células desligadas |
| PPO-1 | −1,9%? (1,2% melhor que Static; 4,8% melhor que Dynamic) | +7,2% / +3,1% vs Static/Dynamic | ≤ 2 UEs | domina as duas baselines |
| DQN | economia > Dynamic (valor não explícito) | > Static/Dynamic | ≤ 1 UE | conservador no custo de ativação |
| PPO-4 | **−19,3%** | **−46,2%** | não reportado | alto custo, alto throughput |
| PPO-5 | **−13,6%** | **−34,0%** | não reportado | idem |

**Leitura crítica dos números.** (i) O texto não reporta intervalos de confiança,
desvio-padrão, número de amostras por CDF nem teste estatístico; as diferenças de
1,2% e 4,8% entre PPO-1 e as baselines estão dentro da variabilidade que os próprios
runs reais locais exibem entre seeds (±20% de throughput, ver `analysis/`). (ii) O
abstract afirma “without compromising stability and performance” e “minimally
impacting the user experience”, mas o PPO-2 sacrifica 52,3% do throughput; a
afirmação é retórica, não suportada pelos números. (iii) A CDF de energia (Fig. 3b)
mostra valores em watts, incompatíveis com a Eq. (1) (R1).

---

## 12. O que o artigo **não** diz (checklist de reprodutibilidade)

| Item | Status |
|---|---|
| Código do xApp / do ambiente Gymnasium | não disponível no artigo; demo promete liberar |
| Dataset (300k pontos) | não disponível |
| Configuração exata do cenário ns-3 | parcial (parâmetros no texto; sem arquivo) |
| Sementes de treino e avaliação | não informadas |
| Split treino/validação/teste | não informado |
| Hiperparâmetros completos de PPO/DQN | não informados (ver §9) |
| Implementação da normalização por quantis | não informada |
| Definição operacional de “energia” em W | não reproduzível (R1) |
| Número de episódios de avaliação | ambíguo (I3) |
| Código das baselines | não disponível; descrição algorítmica parcial |
| Custo computacional / hardware | não informado |
| Tratamento de runs com crash | não discutido |
| Análise estatística | ausente |

---

## 13. Informação explícita × inferência × esclarecimento pendente

| Afirmação | Categoria | Base |
|---|---|---|
| 85 KPMs = 12·7 + 1 | explícita | §IV |
| 128 ações | explícita | §IV |
| 7 células controladas, IDs 2–8 | inferida (N=7, 63 UEs, Fig. 2) | §IV–V |
| “seven additional gNBs” + co-localizado = 8 células | contradição explícita | §V-A vs N=7 |
| γ em watts | **não** sustentada pela Eq. (1) | R1 |
| 500 *data points* = 500 relatórios E2 | inferência | I3 |
| δ é custo de ativação | ambíguo | R6 |
| PPO-1 economiza 1,9% vs Always On | inferência aritmética | §VI |
| 3.000 simulações = ~30M transições? | **não**: 3.000 × 100 = 300k | §V-C |
| Treino offline é “mais seguro” | afirmação normativa, não testada | §IV |

---

## 14. Avaliação crítica de adequação ao CCNC 2025

**Pontos fortes.**
- Problema relevante e bem motivado; arquitetura O-RAN correta em nível conceitual.
- Campanha de dados ambiciosa (~3.000 simulações) e uso de um simulador de pilha
  completa (ns-3 + ns-O-RAN), não de um modelo analítico.
- Comparação com heurísticas da literatura e exploração explícita do trade-off.
- Discussão honesta da limitação de mobilidade (“UEs rarely leave a cell coverage
  area”), que ajuda a explicar por que desligar células “funciona”.

**Fragilidades.**
- **Modelo de energia inconsistente**: a Eq. (1) não produz watts; o simulador local
  não tem modelo de potência de BS. Sem isso, “economizar energia” pode ser apenas
  “transportar menos tráfego” — hipótese que os dados reais locais corroboram
  (energia/throughput *piora* para Static/Dynamic/Random; ver `analysis/`).
- **Recompensa dominada pelo termo `B_sON`** quando a normalização é o fallback
  min-max: 90,9% da penalidade vem de `bs_on`, e os termos de throughput e energia
  praticamente se cancelam (correlação 0,89 entre eles). Isso é medido no dataset
  local (`baselines_dataset.npz`), não no código dos autores, mas a estrutura da
  Eq. (2) e a ausência de calibração documentada tornam o risco estrutural.
- **Treinamento offline do PPO**: o artigo não descreve como o PPO aprende offline.
  No código local, o `OfflineReplayEnv` ignora a ação do agente, o que reduz o PPO a
  *behavior cloning* ponderado pela recompensa. Se o código dos autores fizer o mesmo,
  a comparação PPO vs DQN não isola o algoritmo.
- **Avaliação com 5 seeds (hipótese)** e sem estatística: as conclusões de 1,2%/4,8%
  são frágeis.
- **Artefato não disponível**: sem código/dataset, a reprodutibilidade é baixa.

**Veredito.** O artigo é um bom *proof of concept* de engenharia, mas não estabelece
uma relação causal entre “desligar células” e “economizar energia” no sentido físico,
porque o modelo de energia não é dimensionalmente consistente e o simulador não
modela o desligamento do RF. Essa é a principal lacuna científica a explorar no
mestrado.

---

## 15. Perguntas que precisam de esclarecimento (para os autores ou para o código)

1. Qual é a topologia exata: 7 ou 8 células NR? Onde está a 7ª/8ª?
2. Quais são os 2 KPMs por célula que faltam na enumeração?
3. Como exatamente a energia γ é calculada em watts? Qual modelo de potência?
4. O que exatamente é normalizado por quantis: por célula ou totais da rede? Sobre
   qual conjunto?
5. O termo `B_sON` é subtraído uma vez por passo ou uma vez por célula?
6. δ é um impulso na ativação ou um custo de permanecer ligado?
7. “500 data points” são 500 runs ou 500 relatórios E2?
8. Como o PPO é treinado offline? A ação do agente afeta a transição?
9. Quantas seeds de treino? Como foi feita a seleção de modelo?
10. O código e o dataset serão liberados? Com qual licença?

---

## Referências citadas neste documento

- Bordin et al., *Design and Evaluation of DRL for Energy Saving in Open RAN*, IEEE
  CCNC 2025, DOI `10.1109/CCNC54725.2025.10976108`; arXiv `10.48550/arXiv.2410.14021`.
- Bordin et al., *Demo: Enabling DRL Research for Energy Saving in Open RAN*, IEEE
  CCNC 2025, DOI `10.1109/CCNC54725.2025.10975928`.
- Lacava et al., *ns-O-RAN: Simulating O-RAN 5G Systems in ns-3*, ACM WNS3 2023,
  DOI `10.1145/3592149.3592161`.
- Liu, Natarajan, Xia, *Small Cell Base Station Sleep Strategies for Energy
  Efficiency*, IEEE TVT 65(3), 2016, DOI `10.1109/TVT.2015.2413382`.
- Polese et al., *Improved Handover Through Dual Connectivity in 5G mmWave Mobile
  Networks*, IEEE JSAC 35(9), 2017, DOI `10.1109/JSAC.2017.2720338`.
- López-Pérez et al., *A Survey on 5G RAN Energy Efficiency*, IEEE COMST 24(1), 2022,
  DOI `10.1109/COMST.2022.3142532`.
