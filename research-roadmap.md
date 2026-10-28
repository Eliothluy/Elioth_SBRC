# Roadmap de pesquisa — do estado atual à dissertação

Este roadmap sequencia o trabalho a partir do ponto em que estamos: Fases A
(entendimento do artigo), B (auditoria da implementação) e F (base bibliográfica)
concluídas, com 21 execuções reais de baseline e a análise quantitativa em
`analysis/`. O horizonte é de **12 meses** a contar da aprovação; as datas são
indicativas e todos os prazos são relativos ao portão G0.

## 1. Ponto de partida

| Item | Estado | Evidência |
|---|---|---|
| Entendimento do artigo | concluído | `article-understanding.md` |
| Formulação matemática e ambiguidades R1–R9 | concluído | `mathematical-formulation.md` |
| Auditoria de código (6 divergências materiais) | concluído | `implementation-audit.md` |
| Ambiente, custo e riscos | concluído | `environment-report.md` |
| Base bibliográfica (27 trabalhos, DOIs validados) | concluído | `related-work-matrix.csv`, `literature-search-log.md` |
| Baselines reais (21 runs completos, 3 seeds) | concluído | `analysis/baseline_summary.md` |
| Decomposição do reward | concluído | `analysis/reward_decomposition.md` |
| Dataset à escala do artigo | **ausente** (2.079 transições vs >300.000) | `implementation-audit.md` §4 |
| Agentes DRL comparáveis | **ausentes** (artefactos antigos, 843 transições) | `reproduction-plan.md` |
| Aprovação D1–D5 | **pendente** | `experiment-configs/decisions.template.md` |

## 2. Visão geral

```
Fase 0  Aprovação (D1–D5 + autorização de código)
   │
Fase 1  Robustez: crashes, calibração, logging, testes          ~2 semanas
   │
Fase 2  Reprodução aproximada: baselines 5 seeds + dataset      ~6 semanas
   │
Fase 3  Núcleo de extensões: E2 (algoritmos) + E1/E6 (reward)   ~8 semanas
   │
Fase 4  Validade e fidelidade: E7 (protocolo) + E3/E5           ~8 semanas
   │
Fase 5  Escrita, artefactos, defesa                             ~4 meses
```

As Fases 1–2 são pré-requisito das extensões: sem um dataset calibrado e sem
confiança nos baselines, qualquer resultado de agente é ininterpretável.

## 3. Fase 0 — aprovação (semana 0)

**Objetivo.** Congelar as decisões que mudam o significado dos resultados.

| Passo | Saída |
|---|---|
| Registar D1–D5 em `experiment-configs/decisions.md` | decisões justificadas |
| Autorizar (ou não) edição de `ns-o-ran-gym/` e `ns-3-mmwave-oran/` | escopo de código |
| Autorizar a campanha Tier M (~5 dias de CPU, ~130 GB brutos) | orçamento |
| Confirmar que o resultado será “novo experimento”, não replicação | expectativa |

**Portão G0.** D1–D5 escritos, escopo de código autorizado, orçamento aceite.
Sem G0, nada abaixo começa.

## 4. Fase 1 — robustez da infraestrutura (semanas 1–2)

**Objetivo.** Chegar a 20 runs consecutivos sem crash e com reward não degenerado.

| Atividade | Critério de saída |
|---|---|
| Reproduzir os 3 crashes residuais (`dynamic_2_2_0_3_seed555/556`, `static_2_2_0_3_seed556`) | causa identificada ou *workaround* documentado |
| Correção do bug de serialização LTE/PDCP (se confirmado) | 20 runs seguidos sem abortar |
| `fit_reward_calibration.py` sobre os 21 runs | calibração gravada e versionada |
| Decomposição do reward com a calibração | `bs_on` < 80% da massa negativa; `corr(thr, en)` < 0,85 |
| Testes: suíte rápida + 1 run curto real | 63 testes verdes; run de 1 s completa |
| Política de retenção (apagar traces após ingestão) | 1 run retido em ~5 MB |

**Portão G1.** 20 runs consecutivos sem crash **e** critérios de calibração
cumpridos. Se falhar depois de 2 semanas, ver §12 (contingência).

## 5. Fase 2 — reprodução aproximada (semanas 2–6)

**Objetivo.** Produzir a base de dados e os baselines em 5 seeds que o artigo
exige, dentro do Tier M.

| Semana | Atividade | Saída |
|---|---|---|
| 2–3 | 40 baselines (4 políticas × 5 seeds, tuplas do artigo) | `baseline_summary` com 5 seeds |
| 3–6 | 400 runs de dataset (Always-On 40%, Random 30%, Static/Dynamic 30%) | ≥38.000 transições |
| 5–6 | Conversão + calibração + validação do dataset | `dataset_v1.npz` + *dataset card* |
| 6 | Recalcular Tabela VI local (energia, throughput, RLF) | relatório de reprodução |

**Portão G2.** ≥380 runs utilizáveis, dataset com ≥38.000 transições, reward
calibrado, baselines com 5 seeds e IC. Se falhar, Tier S (181 runs) e redução do
escopo de E2.

## 6. Fase 3 — núcleo de extensões (semanas 7–14)

**Objetivo.** Responder às duas perguntas centrais da tese: *qual algoritmo offline*
e *que reward*.

| Ordem | Extensão | Atividade | Semanas |
|---|---|---|---|
| 1 | **E2** | Implementar/validar BC, CQL, IQL, TD3+BC; 2 rewards × 4 algoritmos × 5 seeds | 7–10 |
| 2 | **E6** | Calibração por componente e por célula; reavaliar ranking | 10 |
| 3 | **E1** | CMDP com restrição de RLF e piso de throughput; 4 braços × 5 seeds | 10–13 |
| 4 | **E7** | Análise de potência, pré-registo, 10 runs de 30 s | 13–14 |

**Portão G3.** Um algoritmo offline válido (não-replay) com avaliação live;
reward com `bs_on` < 80%; pelo menos um braço de E1 que cumpra
`RLF ≤ 2 UEs` **e** `Δthroughput ≥ −10%`. Se G3 falhar, ver §12.

## 7. Fase 4 — validade e fidelidade (semanas 15–22)

**Objetivo.** Responder às objeções “isso é energia?” e “isso é ruído?”.

| Extensão | Atividade | Semanas |
|---|---|---|
| **E3** | Instrumentar ns-3 com `P_0 + Δp·PRB` (EARTH/E3F); recalcular 21 runs; retreinar 1 agente | 15–18 |
| **E5** | Ablation do estado (85 vs 78, global vs por célula) | 18–20 |
| **E4** (opcional) | Custo denso vs impulso + wake-up 71 µs | 20–22 |
| — | Consolidação de figuras e tabelas | 22 |

**Portão G4.** Ranking de políticas estável (Spearman ≥ 0,8) entre proxy e watts,
ou explicação quantificada da diferença; ablation do estado com IC.

## 8. Fase 5 — escrita e defesa (meses 7–12)

| Mês | Atividade | Marco |
|---|---|---|
| 7 | Estrutura da tese + capítulos 1–4 | G5: outline aprovado |
| 8 | Capítulo de reprodução e auditoria | rascunho |
| 9 | Capítulos E2/E1 | artigo 2 submetido |
| 10 | Capítulos E3/E7 + discussão | draft v1 completo |
| 11 | Revisão, artefactos, *reproducibility package* | G6: submissão |
| 12 | Defesa | — |

## 9. Marcos e critérios go/no-go

| Marco | Quando | Critério | Se falhar |
|---|---|---|---|
| **G0** aprovação | semana 0 | D1–D5 + escopo | não começar |
| **G1** robustez | semana 2 | 20 runs sem crash; calibração < 80% | contingência §12.1 |
| **G2** base de dados | semana 6 | ≥380 runs; ≥38k transições | Tier S; E2 com 3 algoritmos |
| **G3** núcleo | semana 14 | algoritmo válido; reward não degenerado; 1 braço QoS viável | §12.2 |
| **G4** fidelidade | semana 22 | ranking estável ou diferença explicada | reportar como limitação |
| **G5** outline | mês 7 | estrutura aprovada | reduzir para 3 capítulos de resultados |
| **G6** submissão | mês 11 | draft completo + artefactos | extensão de prazo |

## 10. Orçamento de recursos

| Recurso | Tier M | Núcleo (E2/E1/E6/E7) | Secundárias (E3/E5/E4) | Total |
|---|---|---|---|---|
| Runs de simulação | 554 | ~80 | ~50 | ~684 |
| Treinos offline | 42 | ~60 | ~25 | ~127 |
| Horas de CPU | ~720 h | ~110 h | ~60 h | ~890 h |
| Dias de parede (6 workers) | ~5,0 | ~1,5 | ~1,0 | ~7,5 |
| Disco retido | ~3 GB | ~0,5 GB | ~0,3 GB | ~4 GB |
| Disco bruto (pico) | ~130 GB | ~20 GB | ~12 GB | ~160 GB |

**Nota.** O pico bruto de 160 GB cabe nos 207 GB livres, mas exige ingestão e
retenção **semanais**; caso contrário o disco enche. O tempo de CPU é ~7,5 dias
distribuídos por 6 meses, compatível com ~20 h/semana de trabalho do estudante.

## 11. Estrutura da dissertação

1. **Introdução** — problema, hipóteses, contribuições.
2. **Fundamentos** — O-RAN/E2, energy saving, DRL, offline RL.
3. **Trabalhos relacionados** — os 27 trabalhos da matriz, com lacunas.
4. **Metodologia** — cenário, estado/acção/reward, simulador, métricas, seeds.
5. **Auditoria e reprodução** — Fases A/B, 21 baselines, decomposição do reward.
6. **Comparação de algoritmos offline** — E2.
7. **Reward com QoS** — E1/E6.
8. **Fidelidade energética e validade** — E3/E7 (e E5, se feita).
9. **Discussão** — ameaças à validade, o que é reprodutível, o que não é.
10. **Conclusão e trabalho futuro** — E4, E8, E9, E10.

**Apêndices.** Configurações, seeds, comandos, *dataset card*, tabela de desvios
R1–R9, instruções de reprodução.

## 12. Contingência

### 12.1 Se G1 falhar (crashes persistentes)

1. Restringir a campanha às políticas que não disparam o bug (Always-On, Static
   4-2-0-1, Dynamic 4-2-0-1) e declarar a limitação.
2. Reduzir `simTime` para 5 s e aumentar o número de seeds.
3. Se nem isso, usar os 21 runs existentes como estudo piloto e mudar o foco da
   tese para a auditoria/reprodutibilidade (E7), que não depende de runs novos.

### 12.2 Se G3 falhar (QoS inviável)

1. Reportar a fronteira de inviabilidade (que RLF/throughput é atingível) — é um
   resultado, não um fracasso.
2. Pivotar E1 para “diagnóstico do reward”: mostrar porque a Eq. (2) não codifica
   QoS e propor uma correção mínima.
3. Manter E2 e E7 como núcleo.

### 12.3 Se o tempo apertar

Ordem de corte: E4 → E5 → E3 → E1 (parcial) → E2 (reduzir a 3 algoritmos).
**Nunca cortar** E7 (validade) nem a reprodução (Fase 2), porque sem elas os
resultados não são interpretáveis.

## 13. Definição de pronto

A tese está pronta quando:

- [ ] Todos os critérios de `reproduction-plan.md` §8.1 são cumpridos.
- [ ] Cada número reportado tem run, seed, comando e data.
- [ ] Nenhum resultado usa dados sintéticos ou avaliação por replay.
- [ ] As 6 divergências da auditoria estão reportadas com impacto estimado.
- [ ] As extensões têm hipótese, desenho, resultado e limitação.
- [ ] O *reproducibility package* reconstrói as figuras principais a partir dos
      `database.db` retidos.
- [ ] A tabela de desvios R1–R9 está no apêndice.

## 14. O que não fazer

- Não treinar com `--synthetic` para “ganhar tempo”: invalida a tese.
- Não comparar agentes com pesos diferentes como se fossem políticas diferentes.
- Não apresentar avaliação offline (`--evaluate` sem `--ns3-path`) como desempenho.
- Não esconder runs abortados: reportá-los e explicar o efeito.
- Não expandir o escopo para E8/E9/E10 antes de G3.
- Não alterar os três repositórios sem autorização explícita (G0).
