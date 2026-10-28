# Frente E1 — Formulação CMDP para o controle de sono de células (design técnico)

**Status:** design aprovado para implementação — **nenhum código foi alterado e nenhum ns-3 foi
executado para este documento.** Todos os números citados provêm de leituras de código-fonte
(caminho:linha) ou de dados reais já coletados (runs do ns-3 em `examples/output/real_runs/`),
listados na §2 e no Apêndice A. Política de dados do `AGENTS.md` respeitada: nenhum dado
sintético, nenhum atalho `--synthetic`.

**Rev. 1 (28/10/2026):** parecer adversarial independente incorporado — ver a seção
**"Revisão e ajustes"** (§8), que corrige o regime do multiplicador de Lagrange (§4.1/§6.3),
torna a Eq. (5.1) obsoleta (substituída pela (5.1′)) e ajusta o protocolo da §6.5. As
medidas novas da revisão foram re-executadas por conta própria sobre os mesmos dados reais
(Apêndice A).

---

## 1. Diagnóstico: por que a soma ponderada da Eq. (2) falha

A recompensa do paper (implementada em `src/nsoran/drl/config.py:324-347` e
`src/environments/es_env.py:887-928`) é

$$
r_t = w_1\,\hat{\rho}_t - w_2\,\hat{\gamma}_t - w_3\,\hat{\varsigma}_t - w_4\,\hat{\delta}_t - w_2\,\hat{B}^{sON}_t ,
\tag{Eq. 2}
$$

onde cada $\hat{\cdot}$ é o componente normalizado por quantis
(`QuantileNormalizer`, `src/environments/es_env.py:346-480`). O problema estrutural é que
**$w_2$ entra duas vezes** e os dois canais que ele pondera ($\hat\gamma_t$ energia e
$\hat B^{sON}_t$ nº de BSs ativas) são *fortemente correlacionados com desligar células*.
Para PPO-2/PPO-3/DQN ($w_2=0{,}40$) a massa negativa máxima é $0{,}8$ contra $0{,}4$ de massa
positiva — razão 2:1 *antes* de qualquer dado. O gradiente da política aprende "desligar
sempre paga duas vezes", e o throughput é o único contra-peso.

Isso é visível nos dados reais (dataset `examples/output/real_runs/baselines_dataset_d1d3.npz`,
componentes normalizados médios por política; cálculo no Apêndice A):

| política | $\hat\rho$ | $\hat\gamma$ | $\hat\varsigma$ | $\hat\delta$ | $\hat B^{sON}$ | PPO-1 ($w$=0,51/0,19/0,20/0,10): $r$ médio | PPO-2 (0,40/0,40/0,10/0,10): $r$ médio | razão neg/pos (PPO-2) |
|---|---|---|---|---|---|---|---|---|
| always_on | 0,622 | 0,609 | 0,445 | 0,583 | 0,921 | −0,121 | −0,466 | 2,87 |
| static    | 0,498 | 0,502 | 0,532 | 0,465 | 0,429 | −0,076 | −0,273 | 2,37 |
| dynamic   | 0,502 | 0,494 | 0,515 | 0,493 | 0,471 | −0,080 | −0,286 | 2,43 |
| random    | 0,378 | 0,399 | 0,434 | 0,527 | 0,334 | −0,086 | −0,238 | 2,57 |

Mesmo no **Always On** (política neutra, sem economia de energia) a recompensa média é
negativa para todos os agentes — o sinal de "economizar" não está calibrado com nenhum ponto
de operação real. Repesar $w_j$ linearmente (manter $\sum w_j = 1$, como exige
`AgentConfig.__post_init__`, `src/nsoran/drl/config.py:193-197`) apenas move o ponto de
equilíbrio ao longo da mesma *trade-off*; não cria uma região de factibilidade. É por isso que
o PPO-2 do paper entrega +24,2% de economia ao custo de −52,3% de throughput: nenhuma escolha
de pesos linear impede a política de "comprar energia com throughput".

**Conclusão E1:** o QoS precisa ser uma **restrição explícita**, não um termo a mais na soma.
Este documento especifica duas formulações de MDP restrito (CMDP):

* **(a) Lagrangiano primal-dual** com restrição explícita de throughput médio
  $\ge \tau_{SLA}$ e/ou RLF $\le \varepsilon$, com o ciclo do multiplicador $\lambda$ dentro do
  treino **offline**;
* **(b) Barreira/limiar ("gate")**: o ganho de energia só entra na recompensa quando o SLA de
  throughput está satisfeito.

Observação de estado atual: a formulação (a) já tem *scaffold* aprovado no repositório —
`src/nsoran/drl/qos.py` (custo *hinge* + `LagrangeMultiplier`), ganchos em
`src/nsoran/drl/rem_dqn.py:423-437,544-546` e `src/nsoran/drl/iql.py:92-102,365-381,496-497` —
mas **não é alcançável pelo CLI** (`src/nsoran/drl/train.py` não expõe flags `qos_*`; os
defaults de `REMDQNConfig`/`IQLConfig` são `None`) e **não existe** para o PPO-SB3
(`src/nsoran/drl/ppo.py` não menciona QoS). A formulação (b) não existe. Este design cobre as
lacunas.

---

## 2. Base de calibração: números reais usados neste design

Fontes (todas reais, já coletadas; nada foi simulado para este documento):

* `examples/output/real_runs/baselines.csv` (modelo de energia *proxy*, legado) e
  `examples/output/real_runs/baselines_d1d3.csv` (modelo EARTH em watts + normalização
  por célula — configuração aprovada em `src/nsoran/drl/real_data.py::EnvironmentModelConfig`).
* Tabelas `grafana` (throughput e RLF brutos **por indicação E2**) dos `database.db` dos runs
  reais referenciados pelo metadata de `baselines_dataset_d1d3.npz` (21 runs: 3 always_on,
  8 static, 7 dynamic, 3 random; 98 indicações/run).

Números de referência (Always On = 3 runs × 99 indicações, `baselines_d1d3.csv`):

| grandeza | always_on | static | dynamic | random |
|---|---|---|---|---|
| throughput médio/indicação (bytes, `QosFlow.PdcpPduVolumeDL_Filter` somado) | **76.686** | 62.196 (−18,9%) | 59.769 (−22,1%) | 31.407 (−59,0%) |
| energia média por indicação (EARTH, W) | 919,47 | 814,83 (**+11,4% economia**) | 823,02 (+10,5%) | 745,10 (+19,0%) |
| RLF médio/indicação (UEs c/ SINR < −5 dB) | 7,20 | 12,43 | 11,52 | 16,30 |
| RLF máx/indicação | 18 | 26 | 27 | 44 |

Percentis por indicação (minha leitura direta do `grafana`; a média AO por indicação dá
77.469 com 98 indicações/run contra 76.686 com 99 no CSV — diferença de −1% pela janela;
uso o valor do CSV como referência canônica por ser o artefato nomeado na frente):

| política | TP p5 | TP p10 | TP p25 | TP p50 | TP p75 | TP p90 | RLF p50 | RLF p75 | RLF p90 | RLF p95 | RLF p99 | RLF máx |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| always_on | 47.400 | 52.260 | 62.263 | 76.178 | 91.625 | 106.915 | 7 | 10 | 12,7 | 14 | 15 | 18 |
| static    | 40.701 | 45.344 | 52.113 | 60.934 | 73.673 | 83.420 | 12 | 15 | 19 | 21 | 24 | 26 |
| dynamic   | 33.514 | 39.467 | 48.732 | 59.767 | 71.590 | 81.878 | 11 | 14 | 17 | 20 | 24 | 27 |
| random    | 0 | 2.076 | 12.776 | 32.092 | 43.717 | 56.169 | 12 | 24 | 33 | 35 | 43 | 44 |

Taxas de violação medidas sob candidatos de $\tau$ (fração de indicações com TP $< \tau$ /
fração de **episódios** com média $< \tau$):

| candidato $\tau$ | AO ind. | static ind. | dynamic ind. | random ind. | AO ep. | static ep. | dynamic ep. | random ep. |
|---|---|---|---|---|---|---|---|---|
| 60%·$\mu_{AO}$ = 46.012 | 4,1% | 10,8% | 18,5% | 79,6% | 0% | 0% | 14% | 100% |
| **70%·$\mu_{AO}$ = 53.680** | 11,6% | 29,5% | 34,1% | 87,1% | 0% | 12% | 14% | 100% |
| 80%·$\mu_{AO}$ = 61.349 | 24,1% | 51,3% | 52,9% | 92,2% | 0% | 38% | 57% | 100% |
| p10 AO = 52.260 | 10,2% | 25,5% | 31,3% | 86,1% | — | — | — | — |

Taxas de violação de RLF por indicação sob candidatos de $\varepsilon$: $\varepsilon=12{,}7$
(p90 AO) → AO 10,2%, static 44,6%, dynamic 39,1%, random 49,0%; $\varepsilon=14$ (p95 AO) →
AO 2,0%, static 28,8%, dynamic 24,3%, random 45,9%; $\varepsilon=15$ (p99 AO) → AO 0,7%,
static 23,2%, dynamic 16,3%, random 42,2%.

Cobertura do dataset offline (fração de transições *factíveis* TP $\ge 53.680$, importante para
o offline: a política só vê viabilidade onde o comportamento a demonstrou): AO 88,4%,
static 70,5%, dynamic 65,9%, random 12,9%.

### 2.1 Escolha recomendada (percentis do Always On, não números inventados)

* **$\tau_{SLA} = 70\%\cdot\mu_{AO} \approx 53.680$ bytes/indicação** (≈ 4,3 Mbit/s
  agregados a 100 ms), com tolerância dual $\delta=0{,}05$ (5% do custo médio admissible).
  Justificativa: é próximo do p10 do próprio AO (52.260), mantém 0% dos episódios AO
  violadores, deixa os baselines "bons" (static/dynamic) *quase* factíveis (12–14% dos
  episódios violam — margem pequena e mensurável para o agente melhorar) e torna o random
  claramente inviável (100%), i.e. a restrição separa exatamente o que o paper chama de
  política aceitável. Alternativa conservadora: 60%·$\mu_{AO}$ = 46.012 (static 0% dos
  episódios viola); alternativa agressiva: 80%·$\mu_{AO}$ (nenhum baseline de referência é
  factível — só usar se o objetivo for superar o static em throughput).
* **$\varepsilon = $ p95 do AO = 14 UEs/indicação** para a restrição por indicação
  (AO viola só 2,0%); ou **$\bar\varepsilon = \mu_{AO} = 7{,}26 \approx 8$ UEs** para a
  restrição em média. Nota de honestidade: o RLF do nosso Always On real é alto em termos
  absolutos (média 7,26, máx 18 em 63 UEs); o valor "máx 2 UEs em RLF" do paper não é
  observável como referência nestes dados — calibrar pelos percentis do AO é o correto.
* **Escalas do custo** (campo já existente em `QoSConstraint`): `throughput_scale =` μ_AO
  = 76.686 (custo adimensional = "fração da referência AO perdida") e `rlf_scale =` 18
  (máx do AO; suaviza o hinge em relação ao default `max(ε,1)`).

---

## 3. Formulação CMDP comum

Estado, ação e dinâmica são os do paper: $s_t \in [0,1]^{85}$, $a_t \in \{0,\dots,127\}$
(bit $i$ = célula $2+i$ ON), episódios de 100 indicações × 100 ms (`DEFAULT_EPISODE_LENGTH`,
`src/nsoran/drl/config.py:150`). Defina, em **unidades cruas** (as mesmas das colunas
`raw_components` do dataset offline, contrato em `src/nsoran/drl/offline.py:19-21`,
ordem `(throughput, energy, rlf, es_on_cost, bs_on)` — `REWARD_COMPONENT_ORDER`,
`src/nsoran/drl/real_data.py:49-55`):

$$
\rho_t = \text{TP}_t \;[\text{bytes/ind}], \qquad
\varsigma_t = \text{RLF}_t \;[\text{UEs/ind}], \qquad
\gamma_t = \text{E}_t \;[\text{W/ind, EARTH}].
$$

O CMDP é

$$
\max_\pi \; J_r(\pi) = \mathbb{E}_\pi\!\left[\sum_{t=0}^{H} \gamma^t r_t\right]
\quad \text{s.a.} \quad
J_c(\pi) = \mathbb{E}_\pi\!\left[\sum_{t=0}^{H} \gamma^t c_t\right] \le \delta ,
\tag{3.1}
$$

com $r_t$ dado pela Eq. (2) (ou pela variante *gate* da §5) e o custo por indicação

$$
c_t \;=\; \underbrace{\frac{\big[\tau_{SLA} - \rho_t\big]_+}{S_\rho}}_{\text{hinge de throughput}}
\;+\; \underbrace{\frac{\big[\varsigma_t - \varepsilon\big]_+}{S_\varsigma}}_{\text{hinge de RLF}},
\qquad [x]_+ = \max(0, x),
\tag{3.2}
$$

exatamente o `QoSConstraint.cost()` de `src/nsoran/drl/qos.py:109-135` (com
$S_\rho=$ `throughput_scale`, $S_\varsigma=$ `rlf_scale`). A restrição "em média" do SLA
($\bar\rho_\pi \ge \tau_{SLA}$ com $\bar\rho_\pi = \mathbb{E}_\pi[\rho_t]$) é o caso
$\varepsilon$-free com hinge linear: $\mathbb{E}[\tau-\rho]_+ = 0 \iff \bar\rho \ge \tau$; o
hinge por transição é o estimador amostrável no buffer offline e é estritamente mais
informativo (penaliza a *cauda* baixa, não só a média).

**Por que $\tau$ em unidades cruas e não no espaço normalizado por quantis:** a normalização
de Eq. (2) é não-paramétrica e recalibrada por arquivo (`RewardNormalizer`,
`src/environments/es_env.py:571-641`); um $\tau$ definido lá mudaria de significado a cada
calibração. O contrato `raw_components` (offline.py:19-21, preenchido por
`real_data.py`) mantém o SLA no domínio físico do ns-3 — decisivo para reportar
"throughput médio ≥ X bytes/ind" na avaliação.

---

## 4. Formulação (a): Lagrangiano primal-dual no treino offline

### 4.1 Objetivo e ciclo do multiplicador

O Lagrangiano do problema (3.1) é $L(\pi,\lambda) = J_r(\pi) - \lambda\, J_c(\pi)$,
$\lambda \ge 0$. No treino offline, o ciclo primal-dual opera **dentro do loop de gradiente**
(ao estilo PDL — cada atualização da política é seguida de um passo de ascensão dual):

$$
\tilde r_t = r_t - \lambda_k\, c_t
\qquad\text{(recompensa modificada usada no alvo TD / no retorno)}
\tag{4.1}
$$

$$
\lambda_{k+1} \;=\; \Pi_{[0,\lambda_{\max}]}\Big( \lambda_k + \eta_\lambda \big( \bar c_k - \delta \big) \Big),
\qquad \bar c_k = \frac{1}{|B_k|}\sum_{t \in B_k} c_t ,
\tag{4.2}
$$

onde $B_k$ é o mini-lote do passo $k$ e $\Pi$ a projeção no intervalo. Equações (4.1)–(4.2)
são exatamente `LagrangeMultiplier.update()` (`src/nsoran/drl/qos.py:205-216`) aplicado em
`REMDQNAgent.update()` (`src/nsoran/drl/rem_dqn.py:544-546`) com a penalização injetada no
alvo TD em `_to_tensors` (`src/nsoran/drl/rem_dqn.py:423-437`).

Hiper-parâmetros recomendados (defaults atuais em `REMDQNConfig`, rem_dqn.py:76-83):
$\eta_\lambda = 0{,}01$; $\lambda_{\max}=100$; $\lambda_0 = 0$; $\delta = 0{,}05$.
Duas correções importantes de design sobre o scaffold atual:

1. **Suavização do dual (recomendado):** com lotes de 256 transições, $\bar c_k$ é ruidoso
   (o hinge é zero em ~88% das transições AO/static). Usar uma média móvel exponencial
   $\tilde c_k = (1-\beta)\bar c_k + \beta \tilde c_{k-1}$ com $\beta = 0{,}9$, ou atualizar
   $\lambda$ a cada $K=100$ passos com a média acumulada. Sem isso o dual oscila e a política
   persegue ruído. *(Mudança local em `LagrangeMultiplier.update`, nova opção `ema`/`period`.)*
2. **O $\lambda\,c$ precisa entrar no bootstrap, não na perda direta.** Para value-based
   (REM-DQN/CQL, IQL) o custo é parte da recompensa e deve ser propagado no alvo
   $y_t = \tilde r_t + \gamma_{dqn}(1-d_t)\max_a Q_{\bar\theta}(s_{t+1},a)$ — o scaffold já
   faz isso corretamente modificando `rewards` antes de computar o alvo (rem_dqn.py:436).
   Qualquer implementação nova (PPO abaixo) deve seguir o mesmo princípio.

> **Rev. 1 (§8.1):** o regime 1:1 com $\eta_\lambda$ constante descrito acima **não converge
> no offline** — o custo médio do lote é uma constante do dataset (média do comportamento),
> não uma função da política. Medido nos dados reais: $\bar c_{dataset} = 0{,}0757 >
> \delta = 0{,}05$ ⇒ deriva dual determinística $+2{,}6\times10^{-4}$/passo, $\lambda
> \approx 12{,}9$ após 50 mil passos e saturação em $\lambda_{\max}=100$ aos ≈388 mil passos
> (dentro do orçamento de 1,2 M do paper). Os ajustes (duas escalas de tempo, crítico de
> custo, grade de λ, novo λ_max) estão na **§8.1** e prevalecem sobre os valores desta seção.

### 4.2 Seleção de checkpoint e parada antecipada

O early-stop atual (menor perda de validação, rem_dqn.py:730-772; maior retorno para PPO,
ppo.py:96-154) **não** sabe nada de factibilidade. Para E1, a regra de seleção do
`best_model` passa a ser lexicográfica (Pareto):

$$
\text{best} = \arg\min_{k \,\big/\, \bar c^{val}_k \le \delta} \; \text{energia estimada}
\quad\text{(na prática: retorno normalizado }\hat J_r\text{ como proxy de energia)},
\tag{4.3}
$$

i.e. entre os checkpoints com custo médio de validação $\le \delta$, escolhe-se o de maior
retorno. Se nenhum checkpoint for factível, reporta-se o de menor $\bar c^{val}$ e o treino é
marcado como "inviável" no `training_summary.json` (campo `qos` já previsto em
`train_iql`, `src/nsoran/drl/iql.py:764-769`).

### 4.3 Ressalva honesta do offline

No treino offline o dual converge para a restrição **sob a distribuição do comportamento**
(behavior policy), não sob a política aprendida: $\bar c_k$ mede violações *registradas*, não
as que a política nova causaria. A factibilidade final só pode ser **atestada na avaliação
live** (§7). O que o Lagrangiano offline garante: (i) o valor-$Q$ e a política são treinados
numa paisagem de recompensa que penaliza estados/ações associados a throughput abaixo de
$\tau$ — o CQL/IQL propaga isso pelos bootstrap; (ii) a cobertura do dataset contém transições
factíveis suficientes (§2: 66–88% das transições static/dynamic/AO têm TP $\ge \tau$
recomendado). Com $\tau = 80\%\cdot\mu_{AO}$ a cobertura cai (~48–76%) e o risco de
extrapolação sobe; com $\tau=60\%$ a restrição quase não discrimina (random já viola pouco em
parte das indicações).

---

## 5. Formulação (b): barreira/limiar — o ganho de energia "portado" pelo SLA

### 5.1 Definição

Aqui o SLA não é um custo somado: é uma **porta** que só deixa o termo de economia entrar na
recompensa quando o throughput da janela respeita o SLA:

$$
r^{gate}_t = w_1 \hat\rho_t - w_3 \hat\varsigma_t - w_4 \hat\delta_t - w_2 \hat B^{sON}_t
\;-\; w_2\,\hat\gamma_t \cdot \mathbb{1}\!\left[\bar\rho^{(W)}_t \ge \tau_{SLA}\right],
\tag{5.1}
$$

onde $\bar\rho^{(W)}_t = \frac{1}{W}\sum_{j=t-W+1}^{t} \rho_j$ é o throughput médio móvel em
unidades cruas sobre as últimas $W$ indicações. Recomenda-se $W = 10$ (1 s de simulação):
a porta instantânea ($W=1$) seria ruidosa demais — o próprio AO tem p5 = 47.400 < 53.680,
ou seja, 11,6% das suas indicações ficariam abaixo de $\tau$ e ele "perderia" o direito de
economizar em nada; com média móvel de 1 s as oscilações do tráfego bursty são filtradas.

> **Rev. 1 (§8.2): a Eq. (5.1) está SUPERSEDA pela (5.1′)** definida na §8.2: (i) o termo
> $-w_2\hat B^{sON}_t$ também é um canal de economia e deve ficar **dentro** da porta;
> (ii) a recompensa passa a depender de $\bar\rho^{(W)}_t \notin s_t$ (não-markoviana para
> um $Q(s,a)$) ⇒ exige variante de estado com a feature da janela; (iii) a esparsidade
> medida da porta dura é de 25–33% nas transições de static/dynamic, o que rebaixa a porta
> dura a ablação e promove a porta suave a variante padrão de (b).

Variantes:

* **Porta dura** (5.1): gradiente da recompensa nulo no lado infeasible. Para PPO/IQL/DQN não
  há gradiente *através* da recompensa (ela entra via retorno/valor), então a porta dura é
  matemática e algoritmicamente válida; o risco é esparsidade de sinal em estados degradados.
* **Porta suave (recomendada como ablação):** substituir a indicadora por
  $\sigma\big((\bar\rho^{(W)}_t - \tau_{SLA})/T\big)$ com temperatura
  $T = 0{,}05\,\mu_{AO} \approx 3.800$ bytes — reduz a variância do retorno em torno do limiar
  sem abandonar a semântica de barreira.
* **Porta composta (extensão opcional):** exigir também $\bar\varsigma^{(W)}_t \le
  \varepsilon$ na condição da indicadora. Só avaliar como ablação; duas portas multiplicam a
  esparsidade.

A vantagem conceitual de (5.1) sobre (4.1): não há hiper-parâmetro dual ($\eta_\lambda$,
$\lambda_{\max}$) e o "preço" do SLA é exato (ou a economia conta, ou não conta). A
desvantagem: não há gradiente que *empurre* a política de volta à factibilidade — se a
política entra numa região com $\bar\rho < \tau$, o termo de energia desaparece e resta só
$w_1\hat\rho$ para puxá-la de volta. Por isso (b) é proposto como **comparação** contra (a),
não substituto.

### 5.2 Interação com a normalização por quantis

A porta usa $\rho$ **cru**; os pesos continuam aplicados aos componentes **normalizados**
(inclusive $\hat\gamma_t$). Não há mudança na calibração dos quantis
(`examples/fit_reward_calibration.py` continua válido como está): a Eq. (5.1) apenas
*zera* o termo $w_2\hat\gamma_t$ em janelas inviáveis. Para o re-scoring offline, a
`reward_components` (normalizadas) **não basta** — a porta precisa de `raw_components`
(coluna 0). Os datasets existentes em `examples/output/real_runs/` foram gerados **antes**
do contrato `raw_components` (verificado: `baselines_dataset_d1d3.npz` não contém o array)
e precisam ser **reconvertidos** — §6.2.

---

## 6. Integração no código (o que muda, arquivo por arquivo)

### 6.1 `compute_reward` (coleta live / env)

* **(a) Lagrangiano:** nenhuma mudança obrigatória na coleta — a restrição vive no treino.
  Recomendação (opcional): gravar o custo por passo na tabela `grafana` (coluna
  `qos_cost`) para diagnóstico pós-run; o env já grava componentes brutos em
  `_last_raw["totals"]` (`src/environments/es_env.py:1588-1594`) e a recompensa em
  `_compute_reward` (es_env.py:1627-1667), então é uma linha de schema + uma chamada a
  `QoSConstraint.cost`.
* **(b) Gate:** nova função em `src/nsoran/drl/config.py` (mantendo o módulo
  dependency-free), p.ex. `compute_gated_reward(...)` recebendo também o $\rho$ cru e o
  estado da janela móvel; `EnergySavingEnv` ganha os parâmetros `cmdp_mode
  ∈ {none, lagrangian, gate}`, `tau_sla`, `gate_window` e mantém o buffer circular de
  $\rho_t$ cru por episódio (reset em `reset()`; o campo `_last_raw["totals"]["throughput"]`
  já fornece o valor por passo). O `info["reward_components"]` passa a incluir o estado da
  porta (`gate_open: bool`) para rastreabilidade.

### 6.2 Dataset offline (`real_data.py` → `offline.py`)

1. **Reconverter os runs reais** com o contrato atual (com `raw_components` e
   `run_labels`), *sem rodar ns-3 de novo* — a conversão lê os run folders existentes:

   ```bash
   cd ns-o-ran-gym
   .venv/bin/python examples/collect_real_dataset.py \
     --runs-dir examples/output/real_runs --agent PPO-2 \
     --output examples/output/real_runs/baselines_dataset_e1.npz   # + run_labels por política
   ```

   (verificar no `--help` do script a flag de labels; `dataset_from_ns3_runs` já aceita
   `run_labels`, `src/nsoran/drl/real_data.py:527-533`, e `infer_throughput_reference`
   (`src/nsoran/drl/qos.py:239-276`) usa os episódios `always_on` para derivar μ_AO —
   preferir isso a hardcodar 76.686.)
2. **Re-scoring:** nova `OfflineDataset.with_cmdp_rewards(mode, constraint, gate)` ao lado de
   `with_rewards_from_components` (`src/nsoran/drl/offline.py:284-315`): `lagrangian`
   computa `c` de `raw_components` (chamando `QoSConstraint.cost`) e **não** aplica $\lambda$
   aqui (o λ muda durante o treino; aplicar no treino), só materializa o array de custos;
   `gate` reescreve `rewards` com a Eq. (5.1) usando a média móvel **por episódio**
   (respeitar `episode_ids`/`timestamps` — a janela não pode vazar entre episódios).

### 6.3 Treino — o que muda por algoritmo

**REM-DQN + CQL (`rem_dqn.py`):** já integrado (§4.1). Faltam: (i) flags no CLI
`src/nsoran/drl/train.py` (`--cmdp-mode`, `--qos-throughput-min/--qos-rlf-max`,
`--qos-throughput-scale/--qos-rlf-scale`, `--qos-lambda-lr/--qos-lambda-max`,
`--qos-tolerance`, e a suavização EMA/periódica do dual); (ii) validação/early-stop com a
regra (4.3) — o `validation_loss` atual (rem_dqn.py:508-516) já reporta `qos_cost_mean`
quando o lote tem `raw_components` (rem_dqn.py:501-505), basta incluí-lo no critério;
(iii) registrar a trajetória de λ (`LagrangeMultiplier.history` já existe,
qos.py:203) no `training_summary.json`.

**PPO (dois caminhos):**

* **IQL discreto (caminho padrão dos PPO-\*, `train.py:92-99,322-359`):** os ganchos são os
  mesmos do REM-DQN e já existem (`iql.py:365-381,496-497,648-650`); faltam as mesmas flags
  de CLI e a regra (4.3) no early-stop do IQL.
* **SB3 PPO sobre `OfflineReplayEnv` (ablação, `ppo.py`):** o PPO consome recompensas do env,
  então o Lagrangiano entra por um **wrapper**:

  ```
  LagrangianReplayWrapper(gym.Wrapper):
      step(action):
          obs, r, term, trunc, info = env.step(action)
          i = info["dataset_index"]                    # já disponível (offline.py:737-751)
          c = constraint.cost(raw_components[i:i+1])[0]
          self._costs.append(c)
          return obs, r - self.lam * c, term, trunc, info
  ```

  e um `BaseCallback` do SB3 que, a cada `n_steps`, faz `lam ← update(mean(self._costs))` e
  loga `qos_cost_mean`/`lambda` (integrável ao `EarlyStopCallback` existente, ppo.py:48-154,
  com o mesmo critério (4.3) sobre o `eval_env` também embrulhado). Ressalva registrada no
  `AGENTS.md` e válida aqui: no replay a ação do agente **não** afeta a transição — o
  Lagrangiano corrige a paisagem de retorno, não a dinâmica; é ablação, não evidência.
  **Rev. 1 (§8.1):** `mean(self._costs)` sofre do mesmo vício da média do comportamento
  (constante do dataset ⇒ deriva positiva de λ); o callback deve usar o estimador por
  episódio de validação da §8.1-A2, ou a grade fixa de λ da §8.1-A3.
  Para o modo `gate`, o wrapper aplica a Eq. (5.1) em vez de subtrair λc (com a janela
  calculada por episódio no dataset).

### 6.4 Avaliação (`evaluate.py` / `compare_runs.py`)

* **Offline (sanity, nunca como resultado de política):** reportar custo médio, taxa de
  violação e $\hat J_r$ por episódio de validação (`QoSConstraint.summary`,
  `qos.py:148-168`, já devolve tudo).
* **Live (evidência real):** a avaliação com `--ns3-path` já extrai os KPIs brutos de
  `info["raw_kpms"]["totals"]` (`src/nsoran/drl/evaluate.py:71` e gotcha documentado no
  `AGENTS.md`). Acrescentar ao relatório por run: throughput médio/indicação vs $\tau_{SLA}$
  (e % de indicações e % de *episódios* factíveis), RLF p95 vs $\varepsilon$, economia de
  energia vs Always On (919,47 W de `baselines_d1d3.csv`), e a curva $\lambda_k$ / estado da
  porta quando aplicável. `examples/compare_runs.py --window 99` continua sendo a
  ferramenta de agregação das campanhas.

### 6.5 Protocolo experimental (para quando a implementação existir)

> **Rev. 1:** passos 2–4 abaixo foram ajustados pela §8 (grade/duas-escalas de λ em vez de
> dual 1:1; porta suave e variante de estado com janela para o modo *gate*; diagnósticos de
> convergência do dual). A numeração original foi mantida; prevalecem os ajustes da §8.3.

1. Reconverter o dataset (§6.2) e conferir μ_AO inferido ≈ 76.686 bytes/ind.
2. Re-treinar offline PPO-2 e DQN com (a) $\tau=53.680$, $\varepsilon=14$, escalas
   μ_AO/18, $\eta_\lambda=0{,}01$ (EMA 0,9), $\lambda_{\max}=100$, $\delta=0{,}05$; e com (b)
   porta dura $W=10$. Seeds fixas; mesmos hiper-parâmetros do paper (só o CMDP muda) para
   atribuição causal limpa.
3. Avaliação live: 5 seeds (555–559), simTime 10 s, `paper_es_use_case.json` — orçamento
   ≈ 90 min/run de wall time (AGENTS.md), ~7,5 h por agente (5 seeds) + baselines já
   coletados. Critério de sucesso E1: economia ≥ static (+11,4%) **com** ≥ 90% dos episódios
   factíveis (média ≥ τ) e RLF p95 ≤ ε; comparar também contra a curva (τ ∈ {60, 70, 80}%).
4. Ablação: porta dura × porta suave ($T=3.800$) × Lagrangiano; e w2-only (sem CMDP) como
   controle.

---

## 7. Riscos e limitações (declarados)

1. **Factibilidade só é atestável live** (§4.3). O dual offline é do comportamento.
2. **Cobertura:** com $\tau$ recomendado, ~12–34% das transições static/dynamic são
   inviáveis e a política deve generalizar "não desligar demais" a partir das viáveis;
   monitorar `feasible_fraction` (`qos.py:166-167`) por política no metadata.
3. **Oscilação do dual** sem suavização (§4.1.1) — validar a trajetória de λ antes de
   confiar no checkpoint.
4. **Porta dura + estados degradados:** sem gradiente de volta à factibilidade (§5.1); se a
   avaliação live mostrar travamento abaixo de τ, migrar para a porta suave ou para o
   Lagrangiano.
5. **RLF absoluto alto no cenário real** (AO médio 7,26): ε calibrado por percentil do AO é
   "não piorar o Always On", não um SLA absoluto de poucos UEs; deixar explícito nos textos.
6. **Janela de 99 × 98 indicações** (CSV × grafana, −1% na média): fixar uma fonte canônica
   (a média móvel do live usa o próprio env; o relatório usa `compare_runs.py --window 99`).
7. **Dual adaptativo no offline não converge como originalmente desenhado** (rev. 1, §8.1):
   a média do lote é constante do comportamento; sem crítico de custo / duas escalas de
   tempo / grade de λ, λ satura e a penalidade domina a Eq. (2).
8. **Porta (b) é não-markoviana e esparsa** (rev. 1, §8.2): a recompensa depende da janela
   $\bar\rho^{(W)} \notin s_t$ e 25–33% das transições de static/dynamic ficam sem sinal de
   energia; mitigação: variante de estado com a feature da janela + porta suave.

---

## 8. Revisão e ajustes (parecer adversarial incorporado)

**Base.** Parecer adversarial independente recebido sobre a versão original deste design.
Todas as alegações **medidas** do parecer foram re-executadas por conta própria, sobre os
mesmos dados reais (runs ns-3 de `examples/output/real_runs/`, tabela `grafana` dos
`database.db`; comandos no Apêndice A) — nenhum ns-3 foi executado e nenhum dado sintético
foi usado. As citações de código do parecer também foram conferidas contra os arquivos lidos.
Nota de transparência: o texto do parecer recebido está **truncado no meio da Q2** (termina
em "`rem_dqn.py:477`"); esta revisão incorpora a Q1 integralmente e, da Q2, os três defeitos
nomeados no seu veredito (esparsidade subestimada, não-markovianidade da janela e
$\hat B^{sON}$ fora da porta). Nada além do que foi nomeado no parecer foi acrescentado.

### 8.1 Q1 — o dual 1:1 com η constante não converge no offline (aceito; defeito estrutural)

**Apontamento aceito.** O loop atual do scaffold dá um passo dual **por passo de gradiente**
(`agent.update(batch)` a cada iteração, `src/nsoran/drl/rem_dqn.py:726-728`; dual em
`rem_dqn.py:544-546`), com $\eta_\lambda=0{,}01$ constante e projeção em $[0,100]$
(`src/nsoran/drl/qos.py:209-214`). Como o `OfflineReplayBuffer` amostra **uniformemente com
reposição** sobre o dataset fixo (`rem_dqn.py:229`), $\mathbb{E}[\bar c_k]$ é a **média do
comportamento** — uma constante que a política aprendida não altera. O dual, portanto, não
persegue ponto sela nenhum.

**Verificação própria (re-executada nesta revisão; τ = 53.680, escala = μ_AO = 76.686):**

| quantidade | valor medido |
|---|---|
| custo hinge médio por política | AO 0,0146 · static 0,0314 · dynamic 0,0504 · random 0,3141 |
| custo médio do dataset misto (21 runs, 2.058 indicações) | **0,0757** > δ = 0,05 |
| deriva dual por passo | $+2{,}57\times10^{-4}$ (λ ← λ + 0,01·(0,0757 − 0,05)) |
| λ após 50 mil passos (exemplo de uso do CLI, `train.py:11`) | ≈ 12,9, monotônico |
| saturação em λ_max = 100 | ≈ 388 mil passos — **dentro** do orçamento de 1,2 M (`config.py:149`) |
| penalidade típica λc em λ = 12,9 (random) | ≈ 4,0; λ·c_max = 9,1 — domina \|r\| ≈ 0,1–0,5 da Eq. (2) (tabela §1) |

Consequência aceita: com λ ≥ 13 o objetivo degenera em "minimizar o hinge do comportamento";
EMA (β=0,9) e atualização a cada K=100 passos (§4.1 original) reduzem a **variância**, não a
**média** — a subida monotônica persiste, apenas mais lisa. O buraco apontado entre a §2.1
(taxas de violação) e a §4.1 (custos hinge) é real: faltava medir $\bar c$ do dataset e
confrontar com δ.

**Ajustes no design (prevalecem sobre a §4.1):**

* **A1 — Regra de coerência δ × $\bar c_D$ (pré-treino, obrigatória):** medir o custo hinge
  médio do dataset de treino antes de qualquer treino com CMDP e registrar no
  `training_summary.json`. Se $\bar c_D \le \delta$, a restrição já é satisfeita pelo
  comportamento (λ → 0, restrição no-op — registrar e não usar CMDP). Se $\bar c_D >
  \delta$ (nosso caso: 0,0757 > 0,05), o dual alimentado por média de lote é **inválido** —
  usar A2 ou A3.
* **A2 — Dual que enxerga a política, em duas escalas de tempo:** (i) treinar um **crítico
  de custo** $V_c(s)$ por TD sobre os mesmos lotes (o custo como "recompensa" do crítico —
  segunda cabeça de valor no REM-DQN/IQL, sem mudar o encoder); (ii) atualizar o dual a cada
  $K_{dual} \ge 100$ passos primais com $\hat J_c(\pi_k)$ estimado **sob a política corrente**
  nos episódios de validação (FQE / Monte-Carlo com o crítico de custo), não com a média do
  lote; (iii) passo Robbins-Monro $\eta_k = \eta_0/(1 + k/K_{dual})^{1/2}$ com teto;
  (iv) critérios de convergência e congelamento: estabilização de λ (variação $< 1\%\cdot
  \lambda_{\max}$ ao longo de 10 atualizações duais ⇒ congelar λ até o fim do treino), proxy
  de gap de dualidade $|\hat J_c^{val}(\pi_k) - \delta|$, e log obrigatório da trajetória
  $\lambda_k$ e da razão penalidade/recompensa.
* **A3 — Alternativa robusta recomendada para o primeiro ciclo: grade fixa de λ.** Treinar
  com λ ∈ {0, 0,5, 1, 2, 5, 10} fixo (um treino por valor, sem dual adaptativo) e escolher,
  pela regra (4.3), o **menor λ cujo custo de validação ≤ δ**. Remove inteiramente a
  questão de convergência do dual ao custo de N treinos offline (baratos perto dos ~90 min
  de wall time por avaliação live). O dual adaptativo (A2) fica como evolução, justificada
  apenas se a grade indicar necessidade de λ intermediário fino.
* **A4 — Guarda de acoplamento:** `qos_lambda_max` default **100 → 10**: com $|r| \le 1$ e
  $c_{max} = 0{,}70$, λ = 10 já dá penalidade máxima ≈ 7 ≫ escala da recompensa; monitorar
  a razão $\lambda\bar c / |\bar r|$ e reduzir λ (ou abortar o treino) se exceder 5.
* **A5 — Wrapper PPO-SB3 (§6.3):** `mean(self._costs)` tem o mesmo vício (média acumulada do
  dataset = constante ⇒ mesma deriva). Substituir pelo estimador por episódio de validação
  (A2) ou pela grade fixa (A3); o callback só loga os diagnósticos.

### 8.2 Q2 — porta (b): esparsidade medida, não-markovianidade e $\hat B^{sON}$ fora da porta (aceito)

* **B1 — Esparsidade medida (re-executada; fração de indicações com $\bar\rho^{(W)}_t <
  \tau = 53.680$, janela completa):**

  | W | always_on | static | dynamic | random |
  |---|---|---|---|---|
  | 1 | 11,6% | 29,5% | 34,1% | 87,1% |
  | 10 | 7,1% | 25,4% | 33,1% | 93,3% |
  | 98 (episódio) | 0,0% | 12,5% | 14,3% | 100,0% |

  A banda medida para static/dynamic é **25–34%** (o parecer reporta 30–37%; a diferença é a
  definição da janela — a ordem de grandeza e a conclusão coincidem). Consequência aceita:
  em ~1/4 a 1/3 das transições das políticas que economizam o termo de energia é zerado, e
  $W=1$ não resolve (ruído maior, esparsidade parecida). **Ajuste:** a porta **suave**
  $\sigma\big((\bar\rho^{(W)}-\tau)/T\big)$, $T = 0{,}05\,\mu_{AO} \approx 3.834$ bytes,
  passa a ser a variante **padrão** de (b); a porta dura vira ablação; a formulação (a)
  permanece a principal do E1 (postura já declarada na §5.1, agora com número que a sustenta).
* **B2 — Não-markovianidade da janela:** $r^{gate}_t$ depende de $\bar\rho^{(W)}_t \notin
  s_t$, mas $Q(s,a)$/$V(s)$ do REM-DQN/IQL/PPO condicionam apenas em $s$ — desvio de
  especificação (o valor aprendido não pode representar a recompensa). **Ajuste:** variante
  de estado E1 com a feature da janela anexada — `full+gate` (86 dims) e `compact+gate`
  (79 dims) — seguindo a maquinaria já existente `STATE_VARIANTS`
  (`src/nsoran/drl/config.py:76-87`), validação pela largura armazenada
  (`src/nsoran/drl/offline.py:130-146`) e o parâmetro `state_variant` do conversor
  (`src/nsoran/drl/real_data.py`). No live, um *wrapper* anexa a feature a partir de
  `info["raw_kpms"]["totals"]["throughput"]` (`src/environments/es_env.py:1588-1594`), sem
  mudar o env, o ns-3, nem o contrato de 85 dims para os agentes não-E1. Histerese com W=1
  foi considerada e **descartada**: o estado interno da porta continuaria fora de $s_t$.
* **B3 — $\hat B^{sON}$ fora da porta:** aceito — $\hat B^{sON}$ é o segundo canal de
  economia multiplicado por $w_2$ e, fora da porta, a política continuaria sendo paga por
  desligar BS mesmo violando o SLA. **Eq. (5.1) SUPERSEDA pela (5.1′):**

  $$
  r^{gate\prime}_t = w_1 \hat\rho_t - w_3 \hat\varsigma_t - w_4 \hat\delta_t
  \;-\; w_2\,\big(\hat\gamma_t + \hat B^{sON}_t\big)\cdot
  g_t, \qquad
  g_t = \sigma\!\Big(\tfrac{\bar\rho^{(W)}_t - \tau_{SLA}}{T}\Big)
  \;\;(\text{dura: } g_t = \mathbb{1}[\bar\rho^{(W)}_t \ge \tau_{SLA}]),
  \tag{5.1′}
  $$

  com $W = 10$ mantido. Nota de escala: no Always On factível ($\hat\gamma = 0{,}609$,
  $\hat B^{sON} = 0{,}921$) a recompensa melhora em $w_2 \cdot 1{,}53$ em relação à (5.1) —
  coerente com a semântica "o ganho de energia só conta com o SLA satisfeito".

### 8.3 Impacto consolidado (o que prevalece)

| seção original | ajuste prevalecente |
|---|---|
| §4.1 (regime dual 1:1, η constante, λ_max=100) | superseded por §8.1-A2/A3 (duas escalas de tempo + crítico de custo, ou grade fixa de λ) e A4 (λ_max=10); EMA/K mantidos apenas como suavização dentro de A2 |
| §5.1 (Eq. 5.1, porta dura como opção principal de (b)) | Eq. (5.1) superseda pela (5.1′); porta suave é a variante padrão; porta dura é ablação |
| §6.2 (re-scoring do modo gate) | passa a exigir também a variante de estado com a janela (B2) e a porta sobre ambos os canais de w2 (B3) |
| §6.3 (wrapper PPO com `mean(self._costs)`) | substituído por §8.1-A5 |
| §6.5 (protocolo) | primeiro ciclo usa **grade fixa de λ** (A3) + diagnósticos de λ obrigatórios nos `training_summary.json`; modo gate usa porta suave + variante de estado com janela |
| §7 (riscos) | riscos 7–8 adicionados |

---

## Apêndice A — Comandos executados para este design (nenhum ns-3)

* Leitura integral de: `src/nsoran/drl/config.py`, `src/environments/es_env.py`,
  `src/nsoran/drl/offline.py`, `src/nsoran/drl/ppo.py`, `src/nsoran/drl/rem_dqn.py`,
  `src/nsoran/drl/qos.py`, `src/nsoran/drl/train.py`, `src/nsoran/drl/real_data.py`
  (trechos citados), `examples/fit_reward_calibration.py`, `examples/output/real_runs/baselines.csv`,
  `examples/output/real_runs/baselines_d1d3.csv`.
* `python` (venv) sobre `baselines_dataset_d1d3.npz` + `database.db` dos runs reais:
  percentis de TP/RLF por política, taxas de violação sob candidatos τ/ε, médias dos
  componentes normalizados e decomposição da Eq. (2) (tabelas da §1 e §2). Os valores
  batem com `baselines.csv` dentro da diferença de janela 99×98.
* **Rev. 1 (§8):** mesmo ambiente (`.venv/bin/python`, somente leitura dos mesmos dados
  reais): (i) custo hinge médio por política e do dataset misto com τ=53.680 e escala
  μ_AO=76.686 (AO 0,0146 · static 0,0314 · dynamic 0,0504 · random 0,3141 · misto 0,0757);
  (ii) dinâmica do dual 1:1 (deriva +2,57·10⁻⁴/passo, λ≈12,9 aos 50 mil passos, saturação
  ≈388 mil passos, penalidade λc até 9,1); (iii) esparsidade da porta por média móvel
  (W ∈ {1, 10, 98}, tabela da §8.2). Saídas conferem com o parecer adversarial (diferença
  apenas na definição da janela da esparsidade, 25–34% vs 30–37%).
* Nenhum teste foi rodado; nenhuma implementação foi feita (escopo do design).
