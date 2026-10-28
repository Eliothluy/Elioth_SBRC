# Projeto e Avaliação de Aprendizado por Reforço Profundo para Economia de Energia em Open RAN

**Título original:** *Design and Evaluation of Deep Reinforcement Learning for Energy Saving in Open RAN*

Matteo Bordin, Andrea Lacava, Michele Polese, Sai Satish, Manoj AnanthaSwamy Nittoor, Rajarajan Sivaraj, Francesca Cuomo, Tommaso Melodia

- Institute for the Wireless Internet of Things, Northeastern University, Boston, MA, EUA
- Sapienza University of Rome, Roma, Itália
- Mavenir, Richardson, TX, EUA

> **Nota:** esta é uma tradução do artigo original. As figuras não foram incorporadas ao arquivo; consulte o PDF original para visualizá-las. As referências bibliográficas foram mantidas no idioma original.

---

## Resumo

Os sistemas sem fio de próxima geração, já amplamente implantados, deverão se tornar ainda mais prevalentes no futuro, representando desafios tanto ambientais quanto econômicos. Este artigo se concentra em melhorar a eficiência energética de sistemas de Rede de Acesso de Rádio Aberta (Open RAN) inteligentes e programáveis por meio da ativação e desativação dinâmicas, em tempo quase real, dos front-ends de radiofrequência (RF) das Estações Base (BS), usando algoritmos de Aprendizado por Reforço Profundo (DRL), especificamente Otimização de Política Proximal (PPO) e Rede Q Profunda (DQN). Esses algoritmos são executados nos Controladores Inteligentes de RAN (RICs), que fazem parte da arquitetura Open RAN, e são projetados para tomar decisões ótimas em nível de rede com base em dados históricos, sem comprometer a estabilidade e o desempenho. Aproveitamos um rico conjunto de Métricas-Chave de Desempenho (KPMs), que servem como estado para o DRL, para criar uma representação abrangente da RAN, juntamente com um conjunto de ações que correspondem a determinado controle exercido no front-end de RF. Estendemos o ns-O-RAN, um simulador realista de código aberto para 5G e Open RAN construído sobre o ns-3, para realizar uma extensa campanha de coleta de dados. Isso nos permite treinar os agentes *offline* com mais de 300.000 pontos de dados e, posteriormente, avaliar o desempenho dos modelos treinados. Os resultados mostram que os agentes de DRL melhoram a eficiência energética adaptando-se às condições da rede, com impacto mínimo na experiência do usuário. Além disso, exploramos o compromisso (*trade-off*) entre vazão (*throughput*) e consumo de energia proporcionado por diferentes projetos de agentes de DRL.

**Termos de indexação:** Open RAN, 5G/6G, Eficiência Energética, Aprendizado por Reforço.

---

## I. Introdução

A Rede de Acesso de Rádio (RAN) na 5ª geração (5G) está se tornando cada vez mais complexa devido à diversidade de casos de uso e à crescente demanda por tráfego de dados.

As redes sem fio de próxima geração enfrentam um duplo desafio: devem atender às crescentes demandas de dados e, ao mesmo tempo, gerenciar o significativo consumo de energia associado à sua operação [1]. À medida que as Estações Base (BSs) 5G são implantadas em densidades sem precedentes para garantir ampla cobertura e conectividade de alta velocidade, a energia necessária para alimentar essa infraestrutura torna-se cada vez mais onerosa. Como resultado, há uma necessidade crítica de soluções inovadoras que possam gerenciar de forma inteligente o consumo de energia em redes 5G, mantendo o desempenho ideal. Atualmente, as BSs RAN tradicionais são baseadas em sistemas de rede monolíticos e hardware proprietário [1]–[3]. Portanto, a intervenção manual é necessária para fins de reconfiguração e planos de economia de energia. A Open RAN, definida pela O-RAN ALLIANCE, rompe com essa abordagem tradicional ao aplicar os princípios de desagregação, abertura, virtualização e programabilidade. Na Open RAN, a estação base (BS) clássica é desagregada: as interfaces entre diferentes nós lógicos são abertas e padronizadas, visando alcançar a interoperabilidade entre múltiplos fornecedores. Além disso, as operações padrão da RAN são virtualizadas e implementadas em hardware de prateleira (*white box*) [4], permitindo reconfigurações dinâmicas da infraestrutura. Essa abordagem baseada em software possibilita a integração de *xApps* — algoritmos projetados para gerenciar a RAN em tempo real. As *xApps* interagem com a RAN por meio da interface E2, viabilizando decisões eficientes de Gerenciamento de Recursos de Rádio (RRM) [5].

A principal contribuição deste artigo é o desenvolvimento de uma *xApp* voltada à redução do consumo de energia da RAN, por meio da ativação e desativação dinâmica dos front-ends de radiofrequência (RF) associados a células específicas, as quais são responsáveis por 65% do consumo de energia da RAN [1]. Ao utilizar análise de dados em tempo real, Aprendizado por Reforço Profundo (*Deep Reinforcement Learning* — DRL) e uma visão global da rede, a *xApp* de economia de energia pode controlar, de forma inteligente, a ativação e desativação de células 5G, otimizando o consumo de energia e, simultaneamente, mantendo os requisitos de Qualidade de Serviço (QoS). Especificamente, projetamos e analisamos diversos modelos de DRL voltados à economia de energia, explorando o compromisso (*trade-off*) entre a economia de energia e a vazão (*throughput*). Outra contribuição deste trabalho é a introdução de funcionalidades de economia de energia no ns-O-RAN — um módulo de código aberto para o simulador ns-3 que modela uma RAN 5G com componentes O-RAN. Essa ferramenta oferece uma plataforma para o desenvolvimento e teste de *xApps*, permitindo uma avaliação de desempenho realista, fundamentada em modelos de canal e protocolos definidos pelo 3GPP [6]. Para o treinamento *offline* dos agentes de DRL, um conjunto de dados (*dataset*) foi gerado por meio de uma campanha de larga escala no ns-3, envolvendo milhares de simulações e quatro diferentes heurísticas de economia de energia encontradas na literatura [3]. Comparamos e testamos diferentes agentes baseados em Otimização de Política Proximal (PPO) e Rede Q Profunda (DQN). Os agentes maximizam uma função de recompensa que combina vazão, consumo de energia, cobertura e um fator de custo associado à ativação e desativação excessivamente frequente de células, visando evitar efeitos de *ping-pong* para os usuários. Nossos resultados mostram a existência de um compromisso (*trade-off*) entre o consumo de energia e a vazão, em que a redução do primeiro pode exigir o sacrifício da segunda. No entanto, em comparação com métodos heurísticos, os agentes de DRL oferecem maior flexibilidade. Eles conseguem se adaptar a condições dinâmicas, melhorando a vazão e, simultaneamente, reduzindo o consumo de energia.

O artigo está estruturado da seguinte forma: a Seção II apresenta o estado da arte; a Seção III descreve a arquitetura do sistema; e a Seção IV aborda o problema de otimização, juntamente com o projeto dos agentes de DRL. A Seção V detalha a configuração experimental, incluindo o cenário no ns-3 e as descrições das abordagens de referência (*baselines*). Por fim, as Seções VI e VII apresentam os resultados obtidos com os modelos de DRL e a conclusão geral.

## II. Estado da Arte

A crescente demanda por dados e as capacidades dos Equipamentos de Usuário (UEs) em redes 5G e além trouxeram preocupações significativas quanto ao consumo de energia, impactando o meio ambiente e os custos operacionais. Isso tem levado a um foco contínuo no aprimoramento da Eficiência Energética (EE) das redes, um tema de constante interesse econômico e social [7]. Em [8], é apresentado um estudo abrangente sobre o assunto, destacando diferentes modelos de consumo de energia e técnicas de EE. Muitos trabalhos, como [9] e [10], concentram-se no desenvolvimento de hardware eficiente para mitigar o impacto dos arranjos de antenas com múltiplas entradas e múltiplas saídas (MIMO) e da Unidade de Rádio (RU) de modo geral. Os autores em [11] revisam técnicas para aumentar os ganhos de EE em sistemas MIMO massivo, incluindo configurações híbridas com ondas milimétricas e redes heterogêneas, ressaltando tanto oportunidades quanto desafios para maximizar a EE em futuras implementações de redes 5G. Os autores de [12] e [13] propõem modelos de consumo de energia baseados em algoritmos de escalonamento inspirados na genética biológica, visando minimizar o consumo energético. Outra técnica específica é o *cell zooming*, apresentada em [14]: ela otimiza o consumo de energia ao ajustar dinamicamente o número de estações base ativas e sua potência de transmissão de acordo com o volume de tráfego, garantindo, ao mesmo tempo, a qualidade de serviço. Outras abordagens partem de soluções desenvolvidas para redes LTE: [15], [16] e [17] demonstram como métodos de economia de energia do LTE — como aqueles que atuam nos domínios de tempo e frequência, ou o mecanismo DRX — podem ser facilmente adaptados para redes 5G. No contexto de economia de energia, a arquitetura Open RAN pode auxiliar na melhoria da escalabilidade da rede, na orquestração e no gerenciamento de recursos de rádio (RRM), contribuindo para uma maior eficiência energética nas implantações da RAN. Apesar dessas promessas, o impacto de uma arquitetura desagregada no consumo de energia ainda é uma questão em aberto e um tópico de interesse. Outras soluções de economia de energia concentram-se mais em abordagens baseadas em software, explorando as possibilidades viabilizadas pelos ciclos de controle inteligente da Open RAN, implementados no Controlador Inteligente da RAN (RIC). Essas soluções utilizam *xApps* para operações em tempo quase real e *rApps* para tarefas periódicas que não exigem tempo real [4]. Em [18], os autores projetam um modelo de consumo de energia e um fluxo de trabalho para a implantação de modelos de aprendizado de máquina, visando minimizar o consumo de energia ao longo de seu ciclo de vida no contexto da Open RAN. Diferentemente de trabalhos anteriores, nosso foco está no desenvolvimento de um agente de DRL para economizar energia na rede, capaz de interagir com a RAN em tempo quase real, por meio de uma estratégia de controle que envolve a ativação e desativação dinâmica do front-end de RF dos Nós B de Próxima Geração (*Next Generation Node Bases* — gNBs). Ao contrário dos métodos existentes, o DRL otimiza decisões com uma visão global da rede, aprendendo e se adaptando a novos padrões sem a necessidade de programação explícita para cada cenário. Por fim, para gerar dados relevantes para alimentar nossos agentes de DRL, implementamos no ns-O-RAN três diferentes linhas de base (*baselines*) fundamentadas em [3], nas quais os autores definem duas políticas de suspensão (*sleeping policies*) estratégicas e uma aleatória, com restrições tanto para os tempos de despertar (*wake-up times*) quanto para as probabilidades de cobertura. Uma descrição das heurísticas implementadas no ns-O-RAN encontra-se na Seção V.

## III. Visão Geral do Sistema

A arquitetura do sistema é apresentada na Fig. 1. Ela consiste em um nó base evoluído LTE (*evolved Node Base* — eNB), $N$ gNBs 5G NR implantados como uma rede não autônoma (*Non-Stand-Alone* — NSA) e um conjunto $U$ de UEs 5G. Os UEs apresentam tipos heterogêneos de tráfego de dados, conforme detalhado na Seção V. Além disso, um RIC de tempo quase real (*Near-Real-Time RIC*) está conectado a cada célula LTE e NR por meio da interface E2. A interface E2 gerencia a troca de dados da rede e as ações de controle do estado das células com periodicidade de tempo quase real. O RIC de tempo quase real é implantado na borda da RAN e inclui uma terminação E2, um coletor de Métricas-Chave de Desempenho (*Key Performance Measurement* — KPM) — que reúne todas as métricas de desempenho das células eNB e gNB — e a *xApp* de economia de energia (*Energy Saving* — ES), responsável por aplicar as ações de controle a cada célula 5G. A *xApp*, em execução no RIC de tempo quase real, gera ações de controle do RIC para o nó E2, determinando o estado ativado ou desativado das células. A opção pelo uso de uma *xApp* — em vez de uma *rApp* — deve-se à necessidade de um mecanismo de controle capaz de responder rapidamente a alterações nas condições da rede, conforme discutido a seguir.

> **Figura 1.** Arquitetura do sistema.

**Controle da RAN para Eficiência Energética e Modos de Suspensão.** A BS pode ter diferentes modos de suspensão, que correspondem à desativação de diferentes partes do nó, desde o amplificador de potência até o front-end de RF completo e até a unidade de processamento em banda base. Diferentes modos de suspensão também são mapeados para diferentes atrasos na desativação e reativação dos componentes da BS [1], [2]. O modo de suspensão 1 é caracterizado por uma duração de desativação mais reativação de 71 microssegundos, e o amplificador de potência, alguns componentes da Unidade de Banda Base (BBU) digital e o front-end analógico podem ser desativados. No modo de suspensão 2, que tem duração de desativação mais reativação de 1 ms, componentes adicionais do front-end analógico podem ser desligados. No modo de suspensão 3, que tem duração mínima de desativação e ativação de 10 ms, a BS pode desativar adicionalmente todo o processamento digital da BBU e quase todo o front-end analógico (exceto o gerador de *clock*). Por fim, no modo de suspensão 4, que tem duração mínima de desativação mais ativação de 1 s, apenas as funcionalidades de despertar são mantidas. A economia de energia realizada neste trabalho refere-se ao modo 1, no qual a desativação e a reativação são executadas o mais rapidamente possível. Isso é particularmente notável porque os equipamentos de radiofrequência (como amplificadores de potência, transceptores e cabos) foram identificados como os maiores consumidores de energia dentro de uma BS, utilizando, tipicamente, cerca de 65% do total de energia da BS [1].

## IV. Projeto do Agente de DRL

Nesta seção, formulamos o problema de otimização para a *xApp* de economia de energia e discutimos o projeto do algoritmo. Optamos por uma técnica de DRL *offline* em vez de métodos *online* por dois motivos principais: as especificações da Open RAN recomendam não treinar modelos de rede *online*, a fim de reduzir o risco de interrupções para os UEs [19]; além disso, o treinamento *online* exige um tempo considerável para a coleta de dados em tempo real e para a interação entre agente e sistema. Embora o aprendizado *offline* não utilize diretamente transições em tempo real, sua metodologia concentra-se na otimização de políticas com base em dados históricos. Detalhes sobre o conjunto de dados de treinamento são apresentados na Seção V.

**Estado.** Para representar o estado da RAN utilizado pelo modelo de DRL, a *xApp* coleta KPMs que são disponibilizadas por meio da interface E2 pelas $N$ estações base NR ou calculadas pela própria *xApp*. As KPMs são funções que representam o estado da RAN no instante $t$ e são selecionadas entre todos os estados disponíveis no sistema devido à sua alta correlação com a função de recompensa. Com um leve abuso de notação, representamos $a_i(t)$ como a ação aplicada no instante $t$ à BS $i \in \{1,\ldots,N\}$, assumindo que o restante da configuração da RAN não mudará. $a_i(t)$ indica se uma BS está ativa ou não no instante $t$. Definimos a vazão (*throughput*) $\rho_i$ como o número de bytes transmitidos na camada PDCP pela célula $i$, e o consumo de energia $\gamma_i$ da célula $i$ como:

$$
\gamma_i\bigl(a_i(t)\bigr) = EC_i\bigl(a_i(t)\bigr) \cdot P_{tx,i}
\tag{1}
$$

em que $EC_i$ e $P_{tx,i}$ são o número total de Unidades de Dados de Pacote (PDUs) transmitidas pela célula $i$ e sua potência de transmissão, respectivamente. Também monitoramos $B_{sON}(a_i(t))$ como o número de BSs ativas no cenário no instante $t$. As 8 KPMs a seguir também são calculadas para cada célula: a razão entre a vazão $\rho_i(\cdot)$ e o consumo de energia $\gamma_i(\cdot)$; o número de UEs em Falha de Link de Rádio (RLF) $\varsigma_i(\cdot)$, ou seja, com Relação Sinal-Interferência mais Ruído (SINR) abaixo de −5 dB; a porcentagem de UEs em RLF em relação ao número total de UEs em $i$; o número e a porcentagem de Blocos de Recursos Físicos (PRBs) agendados; o número de PDUs de Controle de Acesso ao Meio (MAC) com um Esquema de Modulação e Codificação (MCS) que usa 64-QAM; o número de bytes transmitidos na camada física; e o custo $\delta_i(\cdot)$ para ativar uma única célula, calculado como $0{,}9^{0{,}01T_{D_i}}$, em que $T_{D_i}$ é o intervalo de tempo em que a célula permanece ativa, medido em milissegundos. Esta última KPM modela uma função de decaimento exponencial, garantindo que o custo seja maior quando houver um pequeno intervalo de tempo entre as mudanças de ação e menor quando houver um intervalo maior. Um total de 85 KPMs ($12 \cdot N + 1$, com $N = 7$) são coletadas no cenário de simulação descrito na Seção V. Essas KPMs servem como estado para o nosso modelo de DRL, capturando as características essenciais do ambiente que o modelo utiliza para tomar decisões fundamentadas.

**Função de recompensa.** O objetivo do agente é maximizar a vazão dos UEs, encontrando um compromisso ideal entre vazão e consumo de energia e, ao mesmo tempo, evitar RLFs dos UEs individuais e mudanças frequentes de estado em uma mesma célula. Isso é feito definindo a função de recompensa na Eq. (2) como uma combinação ponderada dessas KPMs, com pesos $w_j$, $j \in \{1,\ldots,4\}$, da seguinte forma:

$$
\max_{\mathbf{a}}\ \sum_{t=t_0}^{\infty}\sum_{i=1}^{N}
\Bigl\{ w_1\rho_i(a_i(t)) - w_2\gamma_i(a_i(t)) - w_3\varsigma_i(a_i(t))
- \delta_i(a_i(t))w_4 \Bigr\} - w_2B_{sON}(a_i(t))
\tag{2}
$$

sujeito a

$$
\sum_{j=1}^{k} W_j = 1
\tag{3}
$$

$$
a_i(t) \in \mathcal{A}, \quad \forall t = 0,\ldots,+\infty
\tag{4}
$$

Conforme mostrado na Tabela I, utilizamos diferentes conjuntos de valores para os pesos da recompensa, a fim de atribuir, de forma dinâmica, prioridades distintas a cada componente. A tabela também apresenta a normalização por quantis específica aplicada a cada componente da recompensa. A normalização por quantis transforma os dados em intervalos, em que cada intervalo contém um número igual de limites de intervalo chamados quantis; essa técnica é geralmente aplicada quando os dados disponíveis não seguem uma distribuição gaussiana ou uma distribuição de lei de potência [20]. Por fim, uma normalização padrão, no intervalo $[0,1]$, foi aplicada a cada uma das demais KPMs presentes no vetor de estado.

**Tabela I — Pesos da função de recompensa e estratégia de normalização.**

| Nome  | $w_1$ | $w_2$ | $w_3$ | $w_4$ | Transformação por quantis |
|-------|---------|---------|---------|---------|----------------------------|
| PPO-1 | 0,51    | 0,19    | 0,2     | 0,1     | Normal                     |
| DQN   | 0,4     | 0,4     | 0,1     | 0,1     | Uniforme                   |
| PPO-2 | 0,4     | 0,4     | 0,1     | 0,1     | Uniforme                   |
| PPO-3 | 0,2     | 0,4     | 0,2     | 0,2     | Uniforme                   |
| PPO-4 | 0,4     | 0,32    | 0,18    | 0,1     | Uniforme                   |
| PPO-5 | 0,45    | 0,2     | 0,25    | 0,1     | Normal                     |

**Espaço de ações.** Cada ação corresponde à decisão de ativar ou desativar uma célula específica, com base nas condições da rede e visando maximizar a função de recompensa. Como o ambiente envolve 7 células, o espaço de ações compreende 128 ações possíveis, sendo que a ação do modelo de DRL é representada como um vetor binário de 7 bits.

**Implementação dos agentes de DRL.** Os modelos de DRL foram treinados utilizando dois algoritmos diferentes: PPO e DQN. Para o PPO, utilizamos a implementação do Stable Baselines3 [21] com hiperparâmetros padrão, exceto pelo tamanho do lote (*batch size*) definido como 256 e pelo coeficiente de entropia, ajustado em uma faixa entre 0,001 e 0,003, dependendo dos pesos atribuídos à função de recompensa. O número máximo de passos temporais (*timesteps*) foi definido como 1.200.000, com episódios de 100 passos e uma condição de parada antecipada (*early stop*) caso não houvesse melhoria nas últimas quatro avaliações, após um mínimo de 10.000 passos temporais. O modelo foi treinado em um ambiente personalizado, projetado utilizando a biblioteca Gymnasium [22]. Para o DQN, aproveitamos o projeto de rede neural híbrida de [23], com uma Rede Neural Convolucional (CNN) 1D seguida por três camadas densas de tamanhos [512, 512, 256]. Esses parâmetros de configuração foram escolhidos para equilibrar exploração e explotação durante o treinamento, garantindo um aprendizado estável e eficiente. O processo de treinamento envolve a atualização iterativa dos parâmetros da rede neural para minimizar a discrepância entre os valores Q previstos e os valores Q alvo. Para aumentar a estabilidade e a robustez do aprendizado, utilizamos o algoritmo REM-DQN (*Random Ensemble Mixture*). Além disso, integramos o Aprendizado Q Conservador (*Conservative Q-Learning* — CQL) à nossa estrutura para incentivar uma estimativa conservadora dos valores Q e mitigar o viés de superestimação [24].

## V. Configuração da Simulação

O ambiente de simulação deste artigo é baseado no ns-3, um simulador de código aberto para cenários de rede complexos, que permite a avaliação de desempenho de algoritmos e protocolos de rede. Utilizamos especificamente o módulo ns-O-RAN [6], que modela a RAN e sua conectividade com o RIC de tempo quase real. As linhas de base foram implementadas para conduzir uma extensa coleta de dados, a fim de criar o conjunto de dados de treinamento para os algoritmos de DRL, conforme discutido a seguir.

### A. Cenário de Simulação

> **Figura 2.** Visão geral do cenário de simulação.

Consideramos um cenário urbano denso descrito em [25], com o modelo de canal 3GPP *Urban Micro* (UMi) *Street Canyon* de [26]. Como mostrado na Fig. 2, esse cenário apresenta uma configuração NSA 5G, com um eNB LTE localizado centralmente e um gNB co-localizado com a BS LTE, cercado por sete gNBs adicionais, cada um posicionado a 1.700 metros do centro. Esse cenário é implantado na Faixa de Frequência 1 (FR1), usando uma frequência central de 850 MHz com largura de banda de 20 MHz. O cenário contém 9 UEs para cada BS, totalizando 63 UEs. Implementamos uma estratégia de posicionamento uniforme dos UEs, em que, para cada gNB, 9 UEs são distribuídos aleatoriamente de forma uniforme em um disco centrado na posição do eNB e com raio igual à distância entre sítios (*inter-site distance*) dos gNBs, e uma estratégia de alocação não uniforme, em que $\xi \in \{1,2,3\}$ gNBs são excluídos do posicionamento aleatório. O modelo de mobilidade dos UEs é uma caminhada aleatória bidimensional. Cada UE se move a uma velocidade uniforme entre 2 m/s e 4 m/s.

O tráfego de *downlink* do usuário é uma mistura de quatro modelos de tráfego heterogêneos: 25% dos UEs têm tráfego TCP de *full-buffer*, com taxa de dados de 20 Mbps; outros 25% têm tráfego UDP em rajadas, com taxa média em torno de 20 Mbps, para simular aplicações de *streaming* de vídeo; os 50% finais têm tráfego TCP em rajadas, em que a primeira metade tem taxa média de 750 kb/s e a segunda metade, 150 kb/s. Os modelos de tráfego em rajadas apresentam fases ativa e inativa com duração exponencial aleatória. O tempo de simulação desse cenário é de 10 s, e a periodicidade das ações de controle e das mensagens enviadas ao RIC é de 100 ms.

### B. Funcionalidades de Economia de Energia e Linhas de Base no ns-O-RAN

Para representar adequadamente o problema de otimização descrito neste artigo, estendemos o ns-O-RAN com a possibilidade de ativar e desativar o front-end de uma BS, modelando assim a ação de controle para o caso de uso de economia de energia. Nesta nova implementação, o eNB primário possui um mapa de status dos gNBs, rastreando se uma BS está ativa ou não. Sempre que o agente de DRL na *xApp* gera uma ação de controle, ela é encaminhada ao eNB LTE, que verifica o mapa de status e, se houver alguma atualização, realiza a configuração ou o desligamento do front-end do gNB. Consideramos um mecanismo de *handover* para balancear automaticamente a carga dos UEs sempre que uma BS é ativada ou desativada, com base em [27]. A estratégia para a decisão das células alvo baseia-se em um *Time-to-Trigger* (TTT) dinâmico, que é reduzido proporcionalmente à diferença entre a SINR da célula alvo e a da célula atual.

Também implementamos quatro estratégias de linha de base diferentes para ativar e desativar as BSs e gerar um conjunto de dados de treinamento para os agentes de DRL. A Política Aleatória desativa um número aleatório de BSs a cada segundo, seguindo uma distribuição uniforme. As Políticas Estática e Dinâmica baseiam-se nas heurísticas discutidas em [3]. Ambos os algoritmos determinam os números de BSs a ativar e desativar a cada periodicidade de indicação E2 (ou seja, 100 ms). As heurísticas baseiam-se nos parâmetros $N_{On,P1}$, $N_{On,P2}$ e $N_{Off}$, cuja soma é $N$, o número total de BSs NR no cenário, e são discutidas a seguir. Os algoritmos calculam o número de usuários dentro da área de cobertura de cada BS, levando em consideração uma meta de SINR predefinida (sempre igual a 13 dB) e um raio $R = 2000$ m. As BSs são classificadas com base na maior quantidade de UEs conectados com SINR acima do valor alvo. Os algoritmos então ativam as primeiras $N_{On,P1}$ BSs e não as consideram nas etapas seguintes. A política estática então classifica as BSs restantes com base na menor distância entre elas e o UE mais próximo. A política dinâmica, por sua vez, classifica as BSs com base no tempo que um UE levaria para chegar à localização da BS, incluindo assim também a velocidade do UE na avaliação. Para ambas as políticas, as primeiras $N_{On,P2}$ células são ativadas. Por fim, as $N_{Off}$ células restantes são desativadas. A política Sempre Ativa (*Always On*) mantém todas as BSs ativas durante toda a simulação. Isso representa uma linha de base em que tanto a cobertura quanto o consumo de energia estão em seus níveis máximos.

### C. Conjunto de Dados de Treinamento

O conjunto de dados é composto por aproximadamente 3.000 simulações usando as heurísticas discutidas acima; cada simulação tem 10 s de duração, e os dados das métricas foram salvos a cada 100 ms. Cada variante de heurística foi associada tanto ao posicionamento uniforme quanto ao não uniforme dos UEs para treinamento, enquanto nos testes consideramos apenas UEs alocados uniformemente. Por fim, incluímos diferentes combinações da tupla $(N_{On,P1}, N_{On,P2}, N_{Off})$ para as heurísticas estática e dinâmica, ou seja, (4, 2, 1), (3, 2, 2) e (2, 2, 3).

## VI. Resultados

Nesta seção, relatamos os resultados dos 6 agentes de DRL que foram configurados com os pesos da Tabela I e os parâmetros discutidos na Seção IV. Os resultados baseiam-se em execuções de simulação estatisticamente independentes em relação àquelas do conjunto de dados de treinamento. Nessas simulações, os agentes de DRL realizam inferência *online* e controlam a ativação e a desativação da célula. Os experimentos são repetidos com diferentes sementes, totalizando 500 pontos de dados para cada agente ou linha de base. Na Fig. 3, apresentamos as Funções de Distribuição Acumulada (CDFs) dos valores totais de vazão $\rho$, consumo de energia $\gamma$, número de RLFs por BS e fator de custo $\delta$ associado à ativação de células, definidos na Seção III. As CDFs consideram os valores das KPMs em cada relatório E2. A Fig. 4, por sua vez, considera os valores médios de vazão e consumo de energia e mostra a razão entre ambas as KPMs para cada configuração e aquelas da política Sempre Ativa.

> **Figura 3.** Comparação entre múltiplas KPMs para as linhas de base e os agentes PPO e DQN.
>
> **Figura 4.** Percentual da vazão média em comparação com o percentual do consumo médio de energia.

Os gráficos mostram que a política Sempre Ativa oferece a vazão, o consumo de energia e a cobertura máximos (ou seja, nenhum RLF para os UEs) sem o custo de ativação de células, como esperado. Por outro lado, o agente PPO-2 oferece a melhor economia de energia (Fig. 3b), 24,2% a menos que a política Sempre Ativa. Consequentemente, o número de células desativadas pelo modelo é o maior, resultando em menor vazão (52,3% menor em comparação com a política Sempre Ativa). As linhas de base Estática e Dinâmica permitem que o sistema economize 0,7% e 4,3% mais energia em comparação com a política Sempre Ativa, reduzindo a vazão em 23,5% e 27,6%. A diminuição da vazão está associada a uma menor cobertura, com no máximo 3 UEs em RLF para ambas as linhas de base. Essas linhas de base são conservadoras em termos de ativação e desativação de células, portanto, apresentam baixo custo de ativação. Os agentes DQN e PPO-1 oferecem vazão maior em comparação com as linhas de base Estática e Dinâmica, além de economizarem mais energia que a linha de base Dinâmica. Notavelmente, o agente PPO-1 domina ambas as linhas de base, proporcionando maior economia de energia (melhoria de 1,2% e 4,8%) e maior vazão (7,2% e 3,1%, respectivamente). Ambos os modelos têm cobertura semelhante à das linhas de base, com no máximo 2 e 1 UEs em RLF para PPO-1 e DQN, respectivamente. Em relação ao custo de ativação de células, o DQN adota uma abordagem conservadora para alterar o status de cada BS, superando as linhas de base Estática e Dinâmica. Por fim, os agentes PPO-4 e PPO-5 proporcionam 46,2% e 34% menos vazão, respectivamente, em comparação com a política Sempre Ativa, mas a quantidade de energia economizada é extremamente alta: o primeiro economiza 19,3% e o segundo, 13,6%.

No geral, a Fig. 4 indica que, se o objetivo é economizar energia, uma quantidade de vazão deve ser sacrificada, particularmente em cenários com variabilidade limitada de tráfego de usuários e mobilidade limitada (ou seja, os UEs raramente saem da área de cobertura de uma célula durante as simulações). Nesse caso, cabe ao operador decidir o compromisso ideal com base na intenção específica e nos requisitos de negócio. Por exemplo, o agente PPO-2 pode ser útil em situações em que a economia de energia é a prioridade máxima, enquanto PPO-4 e PPO-5 consomem mais, mas proporcionam maior vazão. Alternativamente, se for necessária economia de energia sem perder muita vazão e cobertura, PPO-1 e DQN são os melhores compromissos entre essas duas métricas investigadas neste artigo.

## VII. Conclusões

Este artigo introduz uma nova abordagem para enfrentar o duplo desafio de atender à crescente demanda de dados das redes 5G e, ao mesmo tempo, gerenciar seu substancial consumo de energia. Propomos uma solução inteligente e programável para redes Open RAN energeticamente eficientes, sem comprometer a qualidade do serviço. A abordagem estudada neste artigo envolve o uso de algoritmos de DRL, especificamente PPO e DQN, para controlar dinamicamente a ativação e a desativação de células 5G com base em dados históricos. Por meio de campanhas abrangentes de simulação e comparação com heurísticas de política, os resultados demonstram a eficácia dos modelos treinados em maximizar uma função de recompensa que engloba vazão, consumo de energia, cobertura e custo de ativação de células. Notavelmente, os resultados mostram que esses modelos oferecem um compromisso entre desempenho e eficiência energética, destacando a necessidade de mecanismos inteligentes de economia de energia em futuras implantações de Open RAN. Ao contribuir para o desenvolvimento de tais mecanismos, o objetivo é melhorar a eficiência e a sustentabilidade das redes 5G diante da crescente demanda de dados e das preocupações ambientais.

## Referências

[1] D. López-Pérez, A. De Domenico, N. Piovesan, G. Xinli, H. Bao, S. Qitao, and M. Debbah, “A Survey on 5G Radio Access Network Energy Efficiency: Massive MIMO, Lean Carrier Design, Sleep Modes, and Machine Learning,” *IEEE Commun. Surv. Tutor.*, vol. 24, no. 1, pp. 653–697, 2022.

[2] F. E. Salem, “Management of advanced sleep modes for energy-efficient 5G networks,” Theses, Institut Polytechnique de Paris, Dec. 2019. [Online]. Available: https://theses.hal.science/tel-02500618

[3] C. Liu, B. Natarajan, and H. Xia, “Small cell base station sleep strategies for energy efficiency,” *IEEE Trans. Veh. Technol.*, vol. 65, no. 3, pp. 1652–1661, 2016.

[4] M. Polese, L. Bonati, S. D’Oro, S. Basagni, and T. Melodia, “Understanding O-RAN: Architecture, Interfaces, Algorithms, Security, and Research Challenges,” *IEEE Commun. Surv. Tutor.*, vol. 25, no. 2, pp. 1376–1411, January 2023.

[5] A. S. Abdalla, P. S. Upadhyaya, V. K. Shah, and V. Marojevic, “Toward Next Generation Open Radio Access Networks: What O-RAN Can and Cannot Do!” *IEEE Network*, vol. 36, no. 6, pp. 206–213, 2022.

[6] A. Lacava, M. Bordin, M. Polese, R. Sivaraj, T. Zugno, F. Cuomo, and T. Melodia, “ns-O-RAN: Simulating O-RAN 5G systems in ns-3,” in *Proceedings of the 2023 Workshop on ns-3*, 2023, pp. 35–44.

[7] S. Buzzi, C.-L. I, T. E. Klein, H. V. Poor, C. Yang, and A. Zappone, “A Survey of Energy-Efficient Techniques for 5G Networks and Challenges Ahead,” *IEEE JSAC*, vol. 34, no. 4, pp. 697–709, 2016.

[8] A. I. Abubakar, O. Onireti, Y. Sambo, L. Zhang, G. K. Ragesh, and M. Ali Imran, “Energy efficiency of open radio access network: A survey,” in *IEEE VTC Spring*, 2023, pp. 1–7.

[9] A. K. Sah, S. K. Singh, S. K. Vankayala, S. Parthasarathy, U. Godavarti, and S. Yoon, “Low-cost Beam-combining Architecture for O-RUs in mmWave Massive MIMO based 5G O-RAN System,” in *IEEE Globecom Workshops (GC Wkshps)*, 2022, pp. 221–226.

[10] Ö. T. Demir, M. Masoudi, E. Björnson, and C. Cavdar, “Cell-Free Massive MIMO in O-RAN: Energy-Aware Joint Orchestration of Cloud, Fronthaul, and Radio Resources,” *IEEE JSAC*, vol. 42, no. 2, pp. 356–372, 2024.

[11] K. N. R. S. V. Prasad, E. Hossain, and V. K. Bhargava, “Energy Efficiency in Massive MIMO-Based 5G Networks: Opportunities and Challenges,” *IEEE Wirel. Commun.*, vol. 24, no. 3, pp. 86–94, 2017.

[12] M. S. A. Shuvo, M. A. R. Munna, S. Sarker, T. Adhikary, M. A. Razzaque, M. M. Hassan, G. Aloi, and G. Fortino, “Energy-efficient scheduling of small cells in 5G: A meta-heuristic approach,” *J. Netw. Comput. Appl.*, vol. 178, p. 102986, 2021.

[13] R. Tan, Y. Shi, Y. Fan, W. Zhu, and T. Wu, “Energy saving technologies and best practices for 5G radio access network,” *IEEE Access*, vol. 10, pp. 51747–51756, 2022.

[14] M. S. Dahal, “Energy saving in 5G mobile communication through traffic driven cell zooming strategy,” *Energy Nexus*, vol. 5, p. 100040, 2022.

[15] Y.-N. R. Li, M. Chen, J. Xu, L. Tian, and K. Huang, “Power saving techniques for 5G and beyond,” *IEEE Access*, vol. 8, pp. 108675–108690, 2020.

[16] T. Kim, Y. Kim, Q. Lin, F. Sun, J. Fu, Y. Kim, A. Papasakellariou, H. Ji, and J. Lee, “Evolution of power saving technologies for 5G new radio,” *IEEE Access*, vol. 8, pp. 198912–198924, 2020.

[17] K.-H. Lin, H.-H. Liu, K.-H. Hu, A. Huang, and H.-Y. Wei, “A survey on DRX mechanism: Device power saving from LTE and 5G new radio to 6G communication systems,” *IEEE Communications Surveys & Tutorials*, vol. 25, no. 1, pp. 156–183, 2023.

[18] T. M. Ho, K.-K. Nguyen, and M. Cheriet, “Energy Efficiency Learning Closed-Loop Controls in O-RAN 5G Network,” in *IEEE Global Communications Conference (GLOBECOM)*, 2023, pp. 2748–2753.

[19] O-RAN Working Group 2, “O-RAN AI/ML Workflow Description and Requirements 1.03,” O-RAN.WG2.AIML-v01.03 Technical Specification, July 2021.

[20] Google Developers. (2022) Machine learning — Prepare Data. [Online]. Available: https://developers.google.com/machine-learning/clustering/prepare-data

[21] A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann, “Stable-baselines3: Reliable reinforcement learning implementations,” *JMLR*, vol. 22, no. 268, pp. 1–8, 2021.

[22] M. Towers, J. K. Terry, A. Kwiatkowski, J. U. Balis, G. d. Cola, T. Deleu, M. Goulão, A. Kallinteris, A. KG, M. Krimmel, R. Perez-Vicente, A. Pierré, S. Schulhoff, J. J. Tai, A. T. J. Shen, and O. G. Younis, “Gymnasium,” Mar. 2023. [Online]. Available: https://zenodo.org/record/8127025

[23] A. Lacava, M. Polese, R. Sivaraj, R. Soundrarajan, B. S. Bhati, T. Singh, T. Zugno, F. Cuomo, and T. Melodia, “Programmable and customized intelligence for traffic steering in 5G networks using open RAN architectures,” *TMC*, vol. 23, no. 4, pp. 2882–2897, April 2024.

[24] A. Kumar, A. Zhou, G. Tucker, and S. Levine, “Conservative Q-Learning for Offline Reinforcement Learning,” in *Advances in Neural Information Processing Systems*, H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, Eds., vol. 33. Curran Associates, Inc., 2020, pp. 1179–1191.

[25] 3GPP, “Study on scenarios and requirements for next generation access technologies,” 3GPP, Technical Release (TR) 38.913, 04 2022, version 17.0.0.

[26] 3GPP, “Study on channel model for frequencies from 0.5 to 100 GHz,” 3GPP, Technical Specification (TS) 38.901, 1 2024, version 17.1.0.

[27] M. Polese, M. Giordani, M. Mezzavilla, S. Rangan, and M. Zorzi, “Improved handover through dual connectivity in 5G mmWave mobile networks,” *IEEE JSAC*, vol. 35, no. 9, pp. 2069–2084, 2017.
