# Auditoria de KPMs do estado de 85 dimensões — Frente E5

**Data:** 2026-10-28 · **Escopo:** análise e proposta de vetor compacto. **Nenhum código-fonte foi alterado** (`implemented=false`).

Estado do paper: `Box(0,1,shape=(85,))` = 12 KPIs × 7 células NR (2..8) + `bs_on` global
(`src/environments/es_env.py:52-72`, `src/nsoran/drl/config.py:24-45`). Layout *cell-major*:
célula *c* (c ∈ {2..8}) ocupa os índices `12·(c−2) .. 12·(c−2)+11`; `bs_on` é o índice 84.

Datasets analisados (**100% reais**, política de dados do `AGENTS.md` — nenhum dado sintético):

| Arquivo | Transições | Episódios | Proveniência (`metadata`) |
|---|---|---|---|
| `examples/output/real_runs/baselines_dataset.npz` | 2.079 | 21 | `source="ns3"`, runs `always_on_seed55x`, `dynamic_*`, `static_*`, `random_*` (seeds 555-557+) |
| `examples/output/real_runs/baselines_dataset_d1d3.npz` | 2.079 | 21 | idem (variantes D1 watts / D3 por célula) |

---

## 1. Redundâncias POR CONSTRUÇÃO (leitura do código)

Os 12 KPIs por célula são definidos em `src/environments/es_env.py:52-65`
(`STATE_FEATURES_PER_CELL`) e espelhados em `src/nsoran/drl/config.py:32-45`
(`CELL_KPM_NAMES`, `KPMS_PER_CELL=12` em `config.py:28`). Três grupos são
redundâncias **exatas** (combinatória linear perfeita) e três são **derivadas**
(funções determinísticas de outros KPIs, não afins):

### Grupo A — cópia exata: `mac_pdu_64qam` ≡ `tb_64qam` (7 colunas)

- **Código:** em `cell_kpis_from_dataframe()`, `src/environments/es_env.py:842` calcula
  `result["tb_64qam"] = tb_64qam.reindex(cell_ids)` e `src/environments/es_env.py:844`
  atribui `result["mac_pdu_64qam"] = tb_64qam.reindex(cell_ids)` — **a mesma série
  `tb_64qam`** (contador `TB.TotNbrDlInitial.64Qam`). O comentário do próprio código já
  admite: `"mac_pdu_64qam", # proxy: TB.TotNbrDlInitial.64Qam` (`es_env.py:62`).
- **Estatística (real):** r = 1,000000 nos 7 pares `[7,9] [19,21] [31,33] [43,45] [55,57]
  [67,69] [79,81]` nos dois datasets; **max|col_norm(i) − col_norm(j)| = 0,0** (bit a bit)
  nos dois arquivos.

### Grupo B — transformação afim exata: `prb_percentage` = `prb_used`/139×100 (7 colunas)

- **Código:** `src/environments/es_env.py:821` —
  `prb_percentage = prb_used / prb_total * 100` com `prb_total=139`.
- **Normalização:** os ranges default são proporcionais — `prb_used: (0, 139)` e
  `prb_percentage: (0, 100)` (`es_env.py:127-128`) —, logo o `MinMaxNormalizer`
  (`es_env.py:494-495`) produz **colunas idênticas** após normalização.
- **Estatística (real):** r = 1,000000 nos 7 pares `[5,6] [17,18] [29,30] [41,42] [53,54]
  [65,66] [77,78]` nos dois datasets; max|diff| = 1,5×10⁻⁸ (erro de arredondamento
  float32) nos dois arquivos.

### Grupo C — dependência linear multi-coluna: `bs_on` = Σ `cell_state` (1 coluna)

- **Código:** `src/environments/es_env.py:874` —
  `bs_on = sum(1 for cell in cell_ids if int(cell_states.get(cell, 1)) == 1)`;
  as 7 colunas `cell_state` vêm do mesmo `states` (`es_env.py:1554`, `es_env.py:866-867`).
- **Normalização:** range de `bs_on` = (0, 7) (`es_env.py:220`) e de `cell_state` = (0, 1)
  (`es_env.py:133`) ⇒ `obs[84] = Σ_c obs[11+12c] / 7` **exatamente**.
- **Estatística (real):** max|obs[84]·7 − Σ cell_state| = 1,8×10⁻⁷ (float32) nos dois
  datasets. É identidade algébrica: nenhuma informação é perdida ao removê-la.

### Grupo D — derivadas determinísticas (redundância parcial; **não cortar** no compacto básico)

| KPI | Definição (código) | Por que manter |
|---|---|---|
| `ee_ratio` | `throughput / max(energy, 1e-9)` — `es_env.py:819-820` | Razão não-afim; no dataset proxy r(thr, ee) = 0,69-0,92; no d1d3 r = 0,989-0,998 (pares `[12,14] [36,38] [48,50] [60,62]`, r 0,9966-0,9981). A interação thr×energia não é recuperável linearmente quando a energia varia (célula OFF). |
| `rlf_percentage` | `rlf_count / total_ues × 100` — `es_env.py:828-832` | O denominador `total_ues` (UEs com SINR reportado, `es_env.py:785-792`) **varia** por timestamp: r(rlf_count, rlf_pct) = 0,70-0,91 — longe do corte 0,995. Carrega a informação do nº efetivo de UEs monitorados, ausente do estado. |
| `energy` (modelo watts/D1) | função determinística de `prb_used` e `cell_state` — `es_env.py:794-818` (EARTH: ON = 130+94·x W, OFF = 75 W) | No d1d3, r(energy, cell_state) = 0,9946-0,9995 (pares `[25,35] [37,47] [49,59] [61,71] [73,83]`); no dataset proxy, r = 0,30-0,65 (sem relação). É derivada do *modelo de energia*, não de outro KPI do estado; é também componente central da recompensa (Eq. 2) e KPI primário no modo proxy. |
| `tb_64qam_ratio` | `tb_64qam / (QPSK+16QAM+64QAM)` — `es_env.py:822-827` | Os contadores QPSK/16QAM **não estão** no estado de 85 dims: a razão adiciona informação (mix de modulação) que não existe em nenhuma outra coluna. De fato não aparece em nenhum par \|r\|≥0,995. |

---

## 2. Correlação com a medição real

Reproduzi a medição fornecida na tarefa com o venv do projeto
(comandos em §5). **Resultado idêntico**: n=2.079 por dataset, `constant_cols` vazio
nos dois, mesmos 14 pares \|r\|≥0,995 em `baselines_dataset.npz` e mesmos 23 pares em
`baselines_dataset_d1d3.npz`, com os mesmos valores de r (ex.: `[12,14]`=0,9976,
`[25,35]`=0,9984, `[61,71]`=0,9995).

Mapeamento de **todos** os pares ≥0,995 (offset = índice mod 12; célula = 2 + ⌊índice/12⌋):

| Par(es) | KPMs | r | Interpretação |
|---|---|---|---|
| 7 pares offset 7↔9 (`[7,9]` etc.) | `tb_64qam` ↔ `mac_pdu_64qam` | 1,0 (ambos) | Grupo A: cópia exata |
| 7 pares offset 5↔6 (`[5,6]` etc.) | `prb_used` ↔ `prb_percentage` | 1,0 (ambos) | Grupo B: afim exata |
| `[12,14] [36,38] [48,50] [60,62]` (só d1d3) | `throughput_dl` ↔ `ee_ratio` | 0,9966-0,9981 | Derivada (Grupo D): nas células sem evento OFF a energia é quase fixa e a razão ≈ afim do throughput |
| `[25,35] [37,47] [49,59] [61,71] [73,83]` (só d1d3) | `energy` ↔ `cell_state` | 0,9984-0,9995 | Modelo EARTH/D1: OFF→75 W constante; a maior parte da variância da energia vem do ON/OFF |
| 14 pares offset 5↔6 e 7↔9 (dataset base) | idem | 1,0 | — |

Ausências informativas (medidas por mim, §5): `rlf_count`↔`rlf_percentage` r = 0,70-0,91;
`throughput`↔`ee_ratio` no dataset base r = 0,69-0,92; `energy`↔`cell_state` no dataset
base r = 0,30-0,65 (energia proxy = `TB.TotNbrDl.1·p_tx`, `es_env.py:794-795`, não depende
do estado ON/OFF — confirma que `baselines_dataset.npz` usa `energy_model="proxy"` e o
`d1d3` usa `energy_model="watts"`).

### Posto linear efetivo (SVD, executado nos dois datasets, obs+next = 4.158 linhas)

| Matriz | rank (tol 1×10⁻⁸) |
|---|---|
| X completo (85 cols) | **70** |
| Compacto 78 (sem `mac_pdu_64qam`) | **70** (ainda 8 colinearidades exatas: 7 `prb_percentage` + `bs_on`) |
| Compacto 70 (proposto) | **70** — **posto completo** |

Número de condição de X₈₅ ≈ 5,3×10²⁰ (`baselines_dataset`) e 4,9×10²⁰ (`d1d3`) —
matriz singular. Os 85 − 70 = 15 graus perdidos são **exatamente** 7×`mac_pdu_64qam` +
7×`prb_percentage` + 1×`bs_on` (Grupos A, B, C), fechando a conta algébrica e numérica.

---

## 3. Vetor compacto proposto: **85 → 70** (a expectativa 85→78 está correta, mas é incompleta)

### Tabela de cortes

| # | KPM removido | Índices (85-dim) | Colunas | Redundância | Evidência código | Evidência estatística (real) |
|---|---|---|---|---|---|---|
| 1 | `mac_pdu_64qam_c{2..8}` | 9, 21, 33, 45, 57, 69, 81 | 7 | cópia exata de `tb_64qam` | `es_env.py:842` vs `es_env.py:844` (mesma série); `es_env.py:62` | r=1,0 (7 pares, 2 datasets); max\|diff\|=0,0 |
| 2 | `prb_percentage_c{2..8}` | 6, 18, 30, 42, 54, 66, 78 | 7 | afim exata de `prb_used` (÷139×100) | `es_env.py:821`; ranges proporcionais `es_env.py:127-128` | r=1,0 (7 pares, 2 datasets); max\|diff\|=1,5×10⁻⁸ |
| 3 | `bs_on` | 84 | 1 | soma exata dos 7 `cell_state` | `es_env.py:874`; ranges (0,7)/(0,1) `es_env.py:220`/`es_env.py:133` | max\|obs[84]·7−Σcell_state\|=1,8×10⁻⁷ |
| | **Total** | 15 índices | **15** | | | rank 85→70, compacto-70 posto completo |

**Dimensão final: 70** = 7 células × 10 KPIs (`throughput_dl, energy, ee_ratio,
rlf_count, rlf_percentage, prb_used, tb_64qam, tb_64qam_ratio, es_on_cost, cell_state`),
sem feature global. Todas as 15 colunas removidas são reconstruíveis sem erro:
`mac_pdu = tb_64qam`; `prb_pct_norm = prb_used_norm`; `bs_on_norm = Σ cell_state / 7`.

### Veredito sobre a expectativa 85→78

- A expectativa da auditoria anterior está **correta como passo mínimo**: a duplicata
  exata `mac_pdu_64qam` justifica sozinha 85−7 = 78, e é o que já está implementado em
  `src/nsoran/drl/config.py:58-72` (`E5_DROPPED_KPMS`, `STATE_DIM_COMPACT=78`,
  variante `"compact"`) e referenciado em `src/nsoran/drl/real_data.py:344` e
  `src/nsoran/drl/offline.py:7`.
- Porém ela está **incompleta**: o rank medido mostra que o compacto-78 ainda carrega
  8 colinearidades perfeitas (7×`prb_percentage` + `bs_on`). O corte sem perda de
  informação linear total é **85→70**. Recomendo registrar `70` como variante alvo da
  E5 (ex.: `STATE_VARIANTS["compact70"]`), mantendo `78` para compatibilidade.
- Nota de rastreabilidade: o comentário `config.py:59-60` cita
  `implementation-audit.md §3.2` como origem do 78, mas esse arquivo **não existe** no
  repositório (`find` sem resultados; este documento o substitui como registro).

### O que NÃO cortar (e por quê)

`rlf_percentage` (r 0,70-0,91 — denominador `total_ues` variável, informação única),
`ee_ratio` (razão não-afim; só colapsa em r≥0,995 nas células sem OFF no d1d3),
`tb_64qam_ratio` (denominadores QPSK/16QAM fora do estado) e `energy` watts
(derivada do modelo EARTH, r 0,995-0,9995 vs `cell_state` no d1d3, mas componente
central da recompensa). Cortes adicionais a essas colunas seriam perda informacional
real, não redundância.

---

## 4. Impacto esperado para IQL/CQL com ~3.400-4.000 transições

Contexto atual: os dois datasets reais somam 4.158 transições (2.079 cada); a faixa
~3.400-4.000 é o orçamento típico de um ciclo de coleta. Efeitos esperados do 85→70:

1. **Razão amostras/dimensão:** N/d sobe de 40,0→48,6 (N=3.400) e de 47,1→57,1
   (N=4.000) — **+21%** de transições por dimensão de entrada, sem coletar nada novo.
2. **Fim da colinearidade perfeita:** X₈₅ é singular (condição ≈ 5×10²⁰; 15 direções
   nulas). Regressores de valor com backup estilo fitted-Q (IQL adv/bc/critic, CQL com
   penalização de conservadorismo sobre Q) ficam com problema de mínimos quadrados
   mal-posto nessas direções: variância teoricamente infinita em OLS e, na prática,
   gradientes redundantes que inflacionam a norma sem acrescentar sinal. O compacto-70
   tem posto completo 70 (medido).
3. **Parâmetros:** a primeira camada densa (largura 512) cai de 85×512 = 43.520 para
   70×512 = 35.840 pesos (−7.680; −17,6%). Para o encoder 1-D CNN do REM-DQN
   (`src/nsoran/drl/networks.py`), o layout por célula passa de 12 para 10 canais,
   mantendo a estrutura convolucional coerente.
4. **Ganho realista:** modesteza esperada — o 78 já remove a pior duplicata (bit a bit
   igual); o ganho incremental 78→70 é o condicionamento (fim das 8 colinearidades
   restantes) e −10,3% de largura. Não substitui a coleta de mais runs reais (política
   de dados proíbe sintéticos; ~50 s de parede por indicação de 100 ms).
5. **Risco:** nenhum para a fidelidade ao paper nas métricas de recompensa — a Eq. 2
   (`es_env.py:921-927`) usa os totais brutos de `info["raw_kpms"]`, não o vetor de
   estado; `bs_on` do reward continua computado em `es_env.py:1593`. O corte é
   invisível para a recompensa e para o `compare_runs.py`.

---

## 5. Comandos executados (reprodutibilidade)

```bash
cd /mnt/legitbet/elioth_sbrc/ns-o-ran-gym
.venv/bin/python /tmp/kpm_audit_analysis.py   # pares |r|>=0.995, rank SVD, condições
.venv/bin/python /tmp/kpm_audit_check2.py     # identidades A/B/C e metadata (source=ns3)
```

Saídas-chave (executadas nesta auditoria): `constant_cols: nenhuma` nos dois datasets;
14 e 23 pares \|r\|≥0,995 (idênticos à medição da tarefa); `rank(X85)=70`,
`rank(compacto78)=70`, `rank(compacto70)=70` (tol 1e-8); max\|obs[84]·7−Σcell_state\|
= 1,8×10⁻⁷; `metadata.source="ns3"`, 21 episódios, run_dirs reais. Testes da suíte do
repositório **não foram executados** (a tarefa não alterou código; `testsExit=null`).

## 6. Fontes

- Código: `src/environments/es_env.py` (linhas 52-65, 62, 121-134, 220, 819-832,
  842-846, 866-867, 874, 794-818, 1554, 1593), `src/nsoran/drl/config.py`
  (linhas 24-45, 58-72), `src/nsoran/drl/real_data.py:344`,
  `src/nsoran/drl/offline.py:7`.
- Dados: `examples/output/real_runs/baselines_dataset.npz`,
  `examples/output/real_runs/baselines_dataset_d1d3.npz` (reais, ns-3).
- Medição de pares fornecida na própria tarefa (reproduzida 1:1 em §2).
