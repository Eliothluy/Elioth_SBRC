# Formulação matemática do problema de energy saving

**Objetivo.** Reconstruir, com precisão simbólica, o problema de decisão formulado em
Bordin et al., *Design and Evaluation of Deep Reinforcement Learning for Energy Saving
in Open RAN* (arXiv:2410.14021, IEEE CCNC 2025), separar o que o texto **define** do
que apenas **sugere**, e mapear cada símbolo para a implementação local. Este documento
é a ponte entre `article-understanding.md` (leitura do artigo) e
`implementation-audit.md` (auditoria de código).

**Regra de leitura.** Toda afirmação recebe um marcador:

- **[D]** definido explicitamente no artigo (com localização);
- **[I]** inferência necessária para fechar o modelo (hipótese declarada);
- **[?]** informação insuficiente — o artigo não permite decidir.

Nenhuma ambiguidade é "corrigida em silêncio". Quando a implementação escolhe uma
leitura, isso é registrado como **divergência de especificação**, não como prova de que
o artigo quis dizer aquilo.

---

## 1. Notação

| Símbolo | Artigo | Significado | Unidade declarada | status |
|---|---|---|---|---|
| $N$ | §IV | número de células NR controladas | — | [D] $N=7$ |
| $i$ | §IV | índice da célula, $i\in\{1,\dots,N\}$ | — | [D] |
| $t$ | §IV | índice do passo de decisão (indicação E2) | — | [D] $t=0,\dots,\infty$ |
| $a_i(t)$ | §IV | ação binária na célula $i$ no passo $t$ | $\{0,1\}$ | [D] |
| $\mathbf{a}(t)$ | §IV | vetor de ações das $N$ células | $\{0,1\}^N$ | [I] |
| $\rho_i$ | §IV | throughput: **bytes transmitidos na camada PDCP** pela célula $i$ | bytes | [D] |
| $\gamma_i$ | Eq. (1) | consumo de energia da célula $i$ | (ver §5) | [D] |
| $EC_i$ | Eq. (1) | número total de PDUs transmitidos pela célula $i$ | PDUs | [D] |
| $P_{tx,i}$ | Eq. (1) | potência de transmissão da célula $i$ | W | [D] |
| $B_{sON}$ | §IV | número de BSs ativas no cenário no passo $t$ | — | [D] |
| $\varsigma_i$ | §IV | número de UEs em RLF na célula $i$ (SINR < −5 dB) | UEs | [D] |
| $\delta_i$ | §IV | "custo para ativar" uma célula: $0.9^{0.01\,T_{D_i}}$ | adimensional | [D] fórmula / [?] semântica |
| $T_{D_i}$ | §IV | intervalo em que a célula permanece **ativa**, em ms | ms | [D] |
| $W_j$ | Eq. (3) | pesos da recompensa, $j=1,\dots,k$ | — | [D] $\sum_j W_j=1$ |
| $w_1..w_4$ | Tabela I | pesos por componente | — | [D] |
| $T$ | §IV | horizonte da soma externa ($t_0 \to \infty$) | passos | [D] $\infty$ |
| $\gamma$ | — | fator de desconto | — | **ausente no artigo** |

**Ponto de notação.** O artigo usa $i$ tanto para o índice da célula
($i\in\{1,\dots,N\}$) quanto, na Eq. (2), dentro de $B_{sON}(a_i(t))$, o que é
inconsistente: $B_{sON}$ é uma quantidade **global**, não indexada por célula. Ver R3
em §6.

---

## 2. O processo de decisão

### 2.1 Estado

O artigo define o estado como um vetor de **85 KPMs** [D, §IV]:

$$
s(t)=\Big(\underbrace{x_{1,1},\dots,x_{1,12}}_{\text{célula }1},\;
\dots,\;
\underbrace{x_{7,1},\dots,x_{7,12}}_{\text{célula }7},\;
\underbrace{B_{sON}}_{\text{global}}\Big)\in\mathbb{R}^{85},
\qquad 12\cdot N+1=85.
$$

Os 12 KPMs por célula, conforme enumerados no texto [D]:

| # | KPM | Símbolo | Observação |
|---|---|---|---|
| 1 | throughput / energia | $\rho_i/\gamma_i$ | razão |
| 2 | número de UEs em RLF | $\varsigma_i$ | SINR < −5 dB |
| 3 | % de UEs em RLF | $\varsigma_i / n_i$ | $n_i$ = UEs na célula |
| 4 | número de PRBs escalonados | — | contador |
| 5 | % de PRBs escalonados | — | contador / total |
| 6 | nº de MAC PDUs com 64-QAM | — | contador |
| 7 | nº de bytes na camada física | — | contador |
| 8 | custo de ativação | $\delta_i$ | $0.9^{0.01T_{D_i}}$ |
| 9 | throughput | $\rho_i$ | implícito (numerador de 1) |
| 10 | energia | $\gamma_i$ | implícito (denominador de 1) |
| — | **2 KPMs não enumerados** | — | **lacuna** |

**Lacuna quantitativa [confirmada].** O texto enumera **8** KPMs "also calculated for
each cell" e afirma 12 por célula. Somando $\rho_i$ e $\gamma_i$ (usados na Eq. 1 e na
razão), chega-se a **10**. Faltam **2** KPMs para fechar $12\cdot 7+1=85$. O artigo não
diz quais são. A implementação local escolheu `tb_64qam_ratio` e `mac_pdu_64qam`
(proxy de `TB.TotNbrDlInitial.64Qam`) — ver `implementation-audit.md`, componente
"Estado 85-d". **Isso é uma decisão de engenharia, não uma recuperação do artigo.**

Formalmente, o estado é **parcialmente especificado**: existe um conjunto
$\mathcal{K}$ de 12 nomes por célula, dos quais 10 são conhecidos. Qualquer
reimplementação que se diga "fiel" precisa declarar sua escolha para os 2 restantes.

### 2.2 Ação

$$
\mathcal{A}=\{0,1\}^{7},\qquad |\mathcal{A}|=2^7=128,
$$

com $a_i(t)=1$ meaning "célula $i$ ativa" e $a_i(t)=0$ "célula $i$ desligada" [D, §IV].
A codificação em inteiro (bit $i$ ↔ célula $i+2$) é uma decisão de implementação
[I], não do artigo.

**Nota sobre o domínio.** O artigo diz "7 cells", mas o cenário tem a eNB LTE
(célula 1) mais as células NR 2–8. As 7 células controladas são as NR; a LTE é
infraestrutura e não entra na ação. Ver I1 em `article-understanding.md`.

### 2.3 Transição

O artigo **não** escreve $P(s'|s,a)$ explicitamente. A transição é o próprio
simulador ns-3: dado $\mathbf{a}(t)$, o xApp envia a decisão, o LTE eNB aplica o
*status map*, dispara handover, e no próximo relatório E2 (100 ms depois) um novo
vetor de KPMs é lido. Formalmente [I]:

$$
s(t+1)\sim P\big(\cdot\mid s(t),\mathbf{a}(t)\big),
$$

com $P$ determinada por: (i) tráfego (4 modelos), (ii) mobilidade random-walk
2–4 m/s, (iii) handover DynamicTtt, (iv) estado RF das células. O processo **não é
Markoviano** em $s$ se os KPMs forem apenas contadores instantâneos — a história
(particularmente $T_{D_i}$ e a fila TCP) importa. O artigo não discute isso.

### 2.4 Recompensa

A Eq. (2) do artigo, transcrita com fidelidade à notação original:

$$
R=\sum_{t=t_0}^{\infty}\sum_{i=1}^{N}\Big\{
w_1\,\rho_i\big(a_i(t)\big)
-w_2\,\gamma_i\big(a_i(t)\big)
-w_3\,\varsigma_i\big(a_i(t)\big)
-w_4\,\delta_i\big(a_i(t)\big)
\Big\}
-w_2\,B_{sON}\big(a_i(t)\big)
\tag{2}
$$

sujeito a

$$
\sum_{j=1}^{k}W_j=1,\qquad a_i(t)\in\mathcal{A}\ \ \forall t.
\tag{3}
$$

Componentes, com a semântica que o texto lhes dá:

| termo | sinal | grandeza | normalização (Tabela I) |
|---|---|---|---|
| $w_1\rho_i$ | $+$ | throughput PDCP | quantil (uniforme ou normal) |
| $-w_2\gamma_i$ | $-$ | energia | quantil |
| $-w_3\varsigma_i$ | $-$ | UEs em RLF | quantil |
| $-w_4\delta_i$ | $-$ | custo de ativação | quantil |
| $-w_2 B_{sON}$ | $-$ | nº de BSs ativas | quantil |

**O que a Eq. (2) não diz:**

1. **Onde termina a soma em $i$.** O termo $-w_2 B_{sON}$ está fora do somatório,
   mas $B_{sON}$ é global. Se ficasse dentro, seria contado 7 vezes. [R3]
2. **Se o mesmo $w_2$ pode ponderar duas grandezas diferentes** (energia e número
   de BSs). A Tabela I soma 1, o que sugere que $w_2$ é uma "fatia" única; usá-lo
   duas vezes viola a leitura ingênua de $\sum W_j=1$. [R4]
3. **Se $\rho_i$ e $\gamma_i$ são por passo ou acumulados.** A soma externa em $t$
   sugere recompensa por passo somada; o código faz isso. [I]
4. **Se há desconto.** Não há $\gamma^t$. Sem desconto e com horizonte infinito, a
   soma diverge para qualquer política com recompensa média positiva — a formulação
   literal é **mal definida**. O código usa o default do SB3 ($\gamma=0.99$), não
   declarado. [R7]

### 2.5 Objetivo

O artigo não escreve o objetivo de otimização de forma explícita. A leitura padrão
[I] é

$$
\pi^\star=\arg\max_{\pi}\ \mathbb{E}_{\pi}\Big[\sum_{t=t_0}^{\infty}\gamma^{t-t_0}R(t)\Big],
$$

com $R(t)$ a Eq. (2) por passo. O artigo treina **offline** (dataset de ~3.000
simulações / >300.000 pontos) e depois avalia, o que é coerente com essa leitura,
mas o texto não define a distribuição de comportamento $\mu(a|s)$ nem a correção
de distribuição — ver §7.

---

## 3. Normalização

O artigo afirma [D, §IV]: "A quantile normalization transforms the data into
intervals where each interval contains an equal number of interval boundaries called
quantiles… Finally, normal normalization in the range [0,1] has been applied to each
remaining KPM in the state vector."

Duas operações distintas são confundidas no texto:

1. **Normalização dos componentes da recompensa** — quantil, com método
   `Normal` ou `Uniform` conforme a Tabela I.
2. **Normalização do estado** — "normal … in [0,1]", ou seja, min-max (ou normal
   CDF), aplicada a cada KPM do vetor de estado.

**O que falta [R9].** O artigo não diz **sobre qual população** os quantis são
ajustados (por célula? pela rede? por passo?), nem se a normalização é aplicada
**antes ou depois** de somar as células. As duas leituras não são equivalentes:

$$
\sum_i n(x_i)\ \neq\ n\!\Big(\sum_i x_i\Big)
$$

para $n$ não linear (quantil, min-max). O código normaliza **totais da rede**
(`_build_raw_kpms` → `"totals"`), o que é uma escolha, não uma dedução.

**Definição operacional usada aqui.** Para uma amostra $x$ e uma referência
$\{x_{(1)},\dots,x_{(M)}\}$ (quantis empíricos ou grade uniforme),

$$
n_{\text{uniform}}(x)=\frac{\#\{x_{(j)}\le x\}}{M},\qquad
n_{\text{normal}}(x)=\Phi^{-1}\!\big(n_{\text{uniform}}(x)\big)\ \text{remapeado para }[0,1].
$$

A implementação local usa `QuantileNormalizer` com `method in {"uniform","normal"}`
e uma grade de 101 pontos quando não há arquivo de calibração. Ver §7.

---

## 4. Análise dimensional

Esta seção é o núcleo da crítica formal: a Eq. (1) **não é dimensionalmente
consistente** com as figuras.

### 4.1 Eq. (1)

$$
\gamma_i(a_i(t))=EC_i(a_i(t))\cdot P_{tx,i}
$$

com $[EC_i]=\text{PDU}$ e $[P_{tx,i}]=\text{W}$. Logo

$$
[\gamma_i]=\text{PDU}\cdot\text{W}.
$$

PDU é adimensional (contagem), então dimensionalmente $[\gamma_i]=\text{W}$. Mas
**numericamente** $\gamma_i$ é "número de PDUs × watts", o que só coincide com
watts se o número de PDUs for interpretado como uma **fração** ou se $EC_i$ for
normalizado. O artigo não diz qual. A Fig. 3b reporta energia em **W** na faixa
4.000–8.000 W, o que é compatível com "W" mas não com "PDU × W" sem uma constante
de conversão. [R1]

**Consequência.** $\gamma_i$ não é reproduzível a partir da Eq. (1) sozinha. É
preciso pelo menos uma de: (a) $EC_i$ é uma taxa (PDU/s), (b) há um fator
$1/\Delta t$, (c) a figura agrega outra coisa.

### 4.2 Throughput

$\rho_i$ é "bytes no PDCP". A Fig. 3a reporta **Mbps**. Converter bytes por
indicação de 100 ms para Mbps exige $\times 8/0.1 = \times 80$. O artigo não
menciona a conversão. [R2]

### 4.3 Custo de ativação

$$
\delta_i=0.9^{\,0.01\,T_{D_i}},\qquad T_{D_i}\ \text{em ms}.
$$

Para $T_{D_i}\ge 0$, $\delta_i\in(0,1]$, **decrescente** em $T_{D_i}$: quanto mais
tempo a célula fica ligada, **menor** o custo. Isso é o oposto de um "custo de
ativação" (que deveria ser um impulso no instante do ON). A Fig. 3d, porém, mostra
$\delta$ até **2,5**, fora do contradomínio da fórmula. [R5, R6]

Duas leituras possíveis, ambas compatíveis com *alguma* parte do texto:

- **L1 (implementada):** $\delta_i$ é um custo de **manter ligado**, que decai com
  o tempo desde o último ON. Célula OFF ⇒ $\delta_i=0$.
- **L2 (literal):** $\delta_i$ é um **bônus/penalidade de ativação** aplicado no
  passo em que $a_i$ passa de 0 para 1, com magnitude dependente do tempo que a
  célula ficou desligada (quanto maior o gap, menor o custo — o que explicaria a
  frase "higher when there is a small time gap between action changes").

A frase do artigo — "ensuring the cost is higher when there is a small time gap
between action changes and lower cost when there is a longer gap" — favorece
**L2**, mas a fórmula com $T_{D_i}$ = "time interval in which the cell remains
active" favorece **L1**. O código implementa L1. **A divergência é material**: em
L2 o custo é esparso (só em transições), em L1 é denso (todo passo ON).

### 4.4 $B_{sON}$

$B_{sON}\in\{0,\dots,7\}$ é uma contagem. Normalizada, vira $\in[0,1]$. Sem
dimensão problemática; o problema é o peso duplicado (R4).

---

## 5. Identificabilidade: o que é possível reproduzir

Separando o que o artigo **fixa** do que **deixa em aberto**:

| Elemento | Fixado? | Consequência |
|---|---|---|
| $N=7$ células NR (IDs 2–8) | sim (com I1) | reproduzível |
| $\lvert\mathcal{A}\rvert=128$, ação binária | sim | reproduzível |
| 85 KPMs = $12N+1$ | sim (contagem) | **nomes de 2 KPMs ausentes** |
| $\rho_i$ = bytes PDCP | sim | conversão para Mbps ausente |
| $\gamma_i=EC_iP_{tx}$ | fórmula sim | **dimensionalmente aberta** |
| $\varsigma_i$ = UEs com SINR < −5 dB | sim | reproduzível |
| $\delta_i=0.9^{0.01T_D}$ | fórmula sim | semântica ambígua |
| Pesos da Tabela I | sim | reproduzível |
| Normalização (Normal/Uniform) | sim | **população não especificada** |
| Horizonte $\infty$, sem desconto | sim | **mal definido** |
| $B_{sON}$ fora do somatório | sim | posição ambígua |
| Transição $P$ | não | depende do ns-3 |

**Veredito formal.** O artigo especifica um MDP **estruturalmente** (dimensões,
ações, componentes) mas **não numericamente** (2 KPMs, unidades de $\rho$ e
$\gamma$, semântica de $\delta$, população de normalização, desconto). Portanto:

- **Reprodução exata do artigo:** impossível sem contato com os autores ou o código
  original. Não há código dos autores disponível (ver `implementation-audit.md`).
- **Reprodução aproximada:** possível, mas exige declarar **9 decisões de
  especificação** (R1–R9) e reportar a sensibilidade a elas.
- **Novo experimento:** qualquer resultado obtido com a implementação local é um
  **novo experimento**, não uma replicação — a menos que se prove equivalência
  paramétrica com o artigo, o que não é possível com os dados públicos.

---

## 6. As nove ambiguidades, formalizadas

| ID | Enunciado formal | Classificação | Efeito quantitativo |
|---|---|---|---|
| **R1** | $[\gamma_i]=\text{PDU}\cdot\text{W}$ vs. Fig. 3b em W | erro confirmado (dimensional) | energia não derivável da Eq. (1) |
| **R2** | $\rho_i$ em bytes vs. Fig. 3a em Mbps | erro confirmado (unidade) | fator 80 ausente |
| **R3** | $-w_2B_{sON}$ fora do $\sum_i$ mas indexado por $i$ | erro confirmado (notação) | 1× vs. 7× por passo |
| **R4** | mesmo $w_2$ em energia e $B_{sON}$ | erro confirmado (provável duplicação) | peso efetivo de energia ≈ $2w_2$ |
| **R5** | $\delta\in(0,1]$ vs. Fig. 3d até 2,5 | informação insuficiente | $\delta$ não calibrado |
| **R6** | "custo de ativar" vs. fórmula decrescente no tempo ligado | hipótese de interpretação | custo denso vs. esparso |
| **R7** | soma até $\infty$ sem desconto | informação insuficiente | objetivo divergente |
| **R8** | sem restrição de cobertura / "todas OFF" | informação insuficiente | política pode zerar a rede |
| **R9** | normalização por quantis sem população definida | informação insuficiente | $\sum n(x_i)\neq n(\sum x_i)$ |

**R4 quantificado.** Se a intenção era $\sum_j W_j=1$ com $W=(w_1,w_2,w_3,w_4)$,
então usar $w_2$ duas vezes dá peso total efetivo

$$
w_1+w_2+w_3+w_4+w_2 = 1+w_2.
$$

Para PPO-2 ($w_2=0.40$), o peso efetivo é **1.40**, e a energia "pesa" 0.80 do total
normalizado — o dobro do que a Tabela I sugere. Isso não é cosmético: ver §7.4.

---

## 7. O que o código efetivamente otimiza

Esta seção mapeia a formulação para `ns-o-ran-gym` e quantifica o desvio. As linhas
citadas são de `src/environments/es_env.py` e `src/nsoran/drl/config.py`.

### 7.1 Estado

`STATE_FEATURES_PER_CELL` (es_env.py:52–65) define 12 nomes:

```
throughput_dl, energy, ee_ratio, rlf_count, rlf_percentage,
prb_used, prb_percentage, tb_64qam, tb_64qam_ratio,
mac_pdu_64qam, es_on_cost, cell_state
```

`build_observation` (es_env.py:686) achata $12\times7$ e acrescenta `bs_on`,
produzindo 85. Normalização: `MinMaxNormalizer` com `_DEFAULT_KPI_RANGES`
(es_env.py:128–140), faixas "generosas" e clip em $[0,1]$.

**Desvios:**

- `energy = TB.TotNbrDl.1 * p_tx` com `p_tx=1.0` (es_env.py:669). Não há watts,
  não há potência estática, não há consumo de RF. É um **proxy de contagem de
  transport blocks**, não a Eq. (1).
- `mac_pdu_64qam` é **o mesmo contador** que `tb_64qam` (es_env.py:658–660).
  Ou seja, dos 12 KPMs, dois são duplicatas — exatamente os dois "extras" que
  faltavam para fechar 12. Isso é uma **escolha de preenchimento**, não uma
  recuperação dos KPMs ausentes.
- `rlf_count` conta UEs únicos com SINR de serving < −5 dB (es_env.py:620–650),
  o que é fiel ao texto, mas "RLF" no 3GPP é um evento de rádio, não um limiar
  instantâneo de SINR. A operacionalização é do artigo, não do padrão.

### 7.2 Recompensa

`compute_reward` (es_env.py:718–733):

```python
reward = (w1*normalized["throughput"]
          - w2*normalized["energy"]
          - w3*normalized["rlf"]
          - w4*normalized["es_on_cost"]
          - w2*normalized["bs_on"])
```

Isso reproduz **literalmente** a Eq. (2) como escrita (incluindo R3/R4), com a
diferença de que a soma em $i$ já foi feita **antes** da normalização
(`_build_raw_kpms` produz `"totals"`). Formalmente o código calcula

$$
R_{\text{code}}(t)=w_1\,n\!\Big(\sum_i\rho_i\Big)-w_2\,n\!\Big(\sum_i\gamma_i\Big)
-w_3\,n\!\Big(\sum_i\varsigma_i\Big)-w_4\,n\!\Big(\sum_i\delta_i\Big)
-w_2\,n(B_{sON}),
$$

enquanto a Eq. (2), lida como soma de termos normalizados por célula, seria

$$
R_{\text{paper}}(t)=\sum_i\Big[w_1n(\rho_i)-w_2n(\gamma_i)-w_3n(\varsigma_i)-w_4n(\delta_i)\Big]-w_2n(B_{sON}).
$$

**Estas duas expressões diferem sempre que $n$ for não linear** — e $n$ é quantil,
portanto não linear. A implementação escolheu a primeira. [R9]

### 7.3 Normalização

`RewardNormalizer` (es_env.py:484–525) usa `QuantileNormalizer` com
`method="uniform"|"normal"` e, sem arquivo de calibração, uma grade uniforme de
101 pontos entre `_DEFAULT_REWARD_RANGES`:

| componente | faixa fallback |
|---|---|
| throughput | 0 – 1e7 |
| energy | 0 – 1e5 |
| rlf | 0 – 63 |
| es_on_cost | 0 – 7 |
| bs_on | 0 – 7 |

**Consequência medida.** Com essas faixas, os componentes reais caem no fundo da
escala. No dataset de baselines (`baselines_dataset.npz`, 2.079 transições, 21
episódios), as médias normalizadas são:

| componente | média normalizada | faixa observada |
|---|---|---|
| throughput | 0.0108 | 0.0050 – 0.0178 |
| energy | 0.0110 | 0.0050 – 0.0169 |
| rlf | 0.1925 | 0.0050 – 0.6964 |
| es_on_cost | 0.1085 | 0.0050 – 0.8960 |
| bs_on | 0.7505 | 0.0050 – 0.9950 |

Ou seja: **throughput e energia são comprimidos a ~1% da escala**, enquanto
`bs_on` ocupa 75%. A normalização não "preserva a informação"; ela **define** o
problema efetivo.

### 7.4 O problema efetivo (medido)

Reconstruindo a recompensa do dataset com os pesos PPO-2/DQN
($w_1=0.40,w_2=0.40,w_3=0.10,w_4=0.10$):

$$
\hat R = 0.40\,n_\rho-0.40\,n_\gamma-0.10\,n_\varsigma-0.10\,n_\delta-0.40\,n_{B_{sON}},
$$

o erro máximo de reconstrução é $3\times10^{-8}$ — confirmando que **esses** são os
pesos usados e que **não houve calibração** (as faixas fallback estão ativas).

Decomposição da recompensa média ($\bar R=-0.3304$):

| termo | contribuição média | % da parte negativa |
|---|---|---|
| $+0.40\,n_\rho$ | $+0.0043$ | — |
| $-0.40\,n_\gamma$ | $-0.0044$ | 1.3% |
| $-0.10\,n_\varsigma$ | $-0.0193$ | 5.8% |
| $-0.10\,n_\delta$ | $-0.0109$ | 3.2% |
| $-0.40\,n_{B_{sON}}$ | $-0.3002$ | **89.7%** |

E as correlações:

$$
\operatorname{corr}(R,\,n_{B_{sON}})=-0.978,\qquad
\operatorname{corr}(n_\rho,\,n_\gamma)=+0.894.
$$

**Leitura.** O agente não está sendo treinado para "economizar energia mantendo
throughput". Está sendo treinado para **minimizar o número de BSs ligadas**, com
um ruído fraco de throughput/energia que quase se cancela. A correlação de 0.894
entre throughput e energia significa que o sinal $w_1n_\rho-w_2n_\gamma$ é
quase nulo — as duas grandezas sobem e descem juntas.

Isso é uma **propriedade do pipeline**, não do artigo. O artigo reporta CDFs de
throughput e energia separadas e escolhe pesos que, com uma calibração adequada,
produziriam um trade-off real. A implementação local, sem calibração, colapsa o
problema em "desligar células". Ver `analysis/baseline_summary.md`.

---

## 8. Três noções de reprodução

Para evitar confusão entre "reproduzir o artigo" e "rodar o código local":

| Noção | Definição | Viável? | O que exigiria |
|---|---|---|---|
| **R-exata** | mesmos números, mesma variância, mesmos hiperparâmetros, mesmo código | não | código dos autores + dataset original + seeds |
| **R-aproximada** | mesma estrutura de MDP, mesmos pesos, mesma escala de cenário; diferenças declaradas | sim, com ressalvas | declarar R1–R9, calibrar recompensa, validar contra as CDFs das Figs. 3–5 |
| **Novo experimento** | usa a infraestrutura local para responder uma pergunta nova | sim | hipótese própria, baseline próprio, sem alegar replicação |

**Critério de decisão.** Só se pode escrever "reproduzimos o artigo" se as curvas
de throughput/energia/RLF ficarem dentro da variabilidade reportada. O artigo
reporta "500 pontos de avaliação por agente"; a implementação local tem 21 runs
completos de 99 indicações. Antes de qualquer alegação, é preciso:

1. calibrar a recompensa com `fit_reward_calibration.py` sobre runs reais;
2. rodar os 6 agentes da Tabela I com ≥3 seeds;
3. comparar CDFs, não médias;
4. reportar a sensibilidade a R1–R9.

---

## 9. Especificação mínima para uma reimplementação fiel

Se o objetivo for construir um MDP **defensável** a partir do artigo, esta é a
especificação mínima que fecha todas as lacunas sem inventar:

```
Estado:  s ∈ R^85
          x_{i,k} para i ∈ {2..8}, k ∈ K, |K| = 12
          K = {ρ_i, γ_i, ρ_i/γ_i, ς_i, ς_i/n_i, PRB_i, PRB_i/PRB_max,
               TB64_i, TB64_i/TB_i, PHY_bytes_i, δ_i, on_i}
          (os 2 KPMs não enumerados: declarar explicitamente)
          s_85 = B_sON = Σ_i on_i

Ação:     a ∈ {0,1}^7, bit j ↔ célula j+2; on=1

Transição: ns-3, passo de 100 ms; T_Di = tempo desde o último on

Recompensa (por passo):
          R = Σ_i [ w1 n(ρ_i) − w2 n(γ_i) − w3 n(ς_i) − w4 n(δ_i) ]
              − w2 n(B_sON)
          com n = quantil ajustado em um conjunto de calibração declarado
          e γ_i em watts (não PDU·W)

Objetivo: Σ_t γ^t R(t), γ declarado (0.99 é o default do SB3, não do artigo)
```

As três escolhas que **precisam** ser declaradas em qualquer publicação derivada:

1. **unidade de $\gamma_i$** (W? PDU? proxy?);
2. **população da normalização** (por célula ou total; conjunto de calibração);
3. **semântica de $\delta_i$** (custo denso de ON ou impulso de ativação).

Sem isso, dois grupos podem implementar "o mesmo artigo" e obter MDPs diferentes.

---

## 10. Resumo executivo

- O artigo define um MDP com $|\mathcal{S}|$ dimensional (85), $|\mathcal{A}|=128$,
  recompensa de 5 componentes e pesos tabelados.
- A formulação é **estruturalmente clara e numericamente incompleta**: 2 KPMs
  ausentes, unidades inconsistentes (R1, R2), peso duplicado (R4), $\delta$
  ambíguo (R5–R6), normalização sem população (R9), horizonte sem desconto (R7).
- A implementação local reproduz a **letra** da Eq. (2), mas: usa um proxy de
  energia, duplica um KPM, normaliza totais e opera com faixas fallback.
- **Medido:** 89.7% do sinal negativo da recompensa vem de `bs_on`; throughput e
  energia se cancelam ($\operatorname{corr}=0.894$). O problema efetivo é
  "minimizar BSs ligadas", não "economizar energia com QoS".
- Consequência: resultados da implementação local são **novo experimento**, não
  replicação. Qualquer comparação com as Figs. 3–5 do artigo exige calibração
  e alinhamento de unidades primeiro.
