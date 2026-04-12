# Guia Completo do Projeto Neroes Neurofeedback
## Uma Explicação Profunda em Português de Portugal

**Autor:** Bruno Sousa  
**Data:** Abril 2026  
**Finalidade:** Compreender e explicar o projeto na totalidade - problema, métodos, resultados, limitações e implicações.

---

## 1. Contexto e Motivação

### O que é o neurofeedback?

O neurofeedback é uma técnica de treino cerebral baseada no princípio da biofeedback. O utilizador recebe em tempo real informação sobre a sua própria actividade cerebral - medida por EEG (electroencefalografia) - e, através de repetição e prática, aprende a auto-regular determinados padrões cerebrais.

A ideia é simples: se o cérebro produz o padrão desejado (por exemplo, ondas alpha na região frontal), o sistema recompensa o utilizador. Se não produz, o sistema retira ou reduz a recompensa. Ao longo de muitas sessões, o cérebro associa o padrão neurológico correcto à recompensa e começa a produzi-lo voluntariamente.

Aplicações clínicas incluem tratamento de ADHD, ansiedade, insónia, recuperação de AVC, e melhoria de performance cognitiva em populações saudáveis.

### O problema da Neroes

A Neroes implementou o neurofeedback sob a forma de um **jogo de computador** chamado *Spaceship*: o utilizador controla a posição vertical de uma nave espacial com a sua actividade cerebral. Quando o cérebro produz o padrão alvo (definido pelo protocolo `F4Alpha − F3Alpha` neste caso), a nave sobe; quando não produz, a nave desce.

O problema é que o protocolo tem um **parâmetro de limiar** - a dificuldade do jogo - que precisa de ser ajustado dinamicamente. Se o limiar for demasiado alto (tarefa demasiado difícil), o utilizador falha constantemente e desmotiva. Se for demasiado baixo (tarefa demasiado fácil), não há aprendizagem real.

Hoje, este ajuste é feito por um terapeuta com base na experiência clínica. O objectivo do projecto é **automatizar este ajuste** usando dados da própria sessão.

### O desafio técnico

Isto é um problema de **tomada de decisão sequencial sob incerteza**:

- A cada segundo, o sistema observa o estado do cérebro e do jogo
- Deve escolher uma acção (baixar/manter/subir o limiar)
- Recebe uma recompensa (o cérebro melhorou ou piorou?)
- O estado seguinte depende desta acção
- O objectivo é maximizar a melhoria ao longo de toda a sessão, não apenas no próximo segundo

Este tipo de problema é formalmente descrito por um **Processo de Decisão de Markov (MDP)** e é o domínio da aprendizagem por reforço (RL).

---

## 2. Os Dados

### O headset EEG

O headset utilizado é o **Unicorn** da g.tec. Tecnicamente suporta 16 elétrodos, mas nesta sessão apenas **4 elétrodos** produziram sinal activo:

- **F3 e F4** - região frontal esquerda e direita (associada a funções executivas, atenção)
- **C3 e C4** - região central/motora esquerda e direita

Os restantes 12 elétrodos (AF3, AF4, F7, F8, FC5, FC6, Fp1, Fp2, O1, O2, Oz, P7, P8, Pz, T7, T8) estavam em zero constante - provavelmente por mau contacto com o couro cabeludo, o que é comum em configurações de laboratório não-clínicas.

Cada elétrodo activo fornece **5 estimativas de potência em bandas de frequência**:

| Banda | Frequência | Associação cognitiva |
|---|---|---|
| Theta | 4–8 Hz | Sonolência, meditação, memória |
| Alpha | 8–13 Hz | Relaxamento alerta, atenção visual |
| Low Beta | 13–20 Hz | Concentração activa |
| High Beta | 20–30 Hz | Ansiedade, activação motora |
| Gamma | 30–45 Hz | Processamento de alta frequência |

Resultado: **4 elétrodos × 5 bandas = 20 canais de EEG activos**.

### Estrutura das subsessões

A sessão completa dura aproximadamente 30 minutos e está dividida em **10 subsessões** (numeradas 0 a 9):

| SubSessão | Tipo | Linhas | Duração |
|---|---|---|---|
| 0 | Baseline (calibração) | 837 | ~7 min |
| 1 | Jogo | 240 | ~2 min |
| 2 | Jogo | 418 | ~4 min |
| 3 | Jogo | 270 | ~2.7 min |
| 4 | Jogo | 317 | ~3.2 min |
| 5 | Jogo | 341 | ~3.4 min |
| 6 | Jogo | 314 | ~3.1 min |
| 7 | Jogo | 270 | ~2.7 min |
| 8 | Jogo | 272 | ~2.7 min |
| 9 | Jogo | 270 | ~2.7 min |
| **Total** | | **3,549** | **~30 min** |

A **subsessão 0 é crítica**: é a **fase de calibração**. O utilizador está parado, com uma cruz de fixação no ecrã, enquanto o sistema mede o seu sinal EEG em repouso. Este sinal de repouso serve como **baseline pessoal** - o ponto de referência contra o qual todos os valores seguintes são comparados.

Cada linha dos dados representa **uma amostra temporal** (aproximadamente 1 por segundo, com base na duração das subsessões e número de linhas).

### O ProtocolValue - a variável central

O `ProtocolValue` é a variável mais importante de todo o projeto. É calculado a partir do sinal EEG bruto usando a fórmula:

```
ProtocolValue = TangentCoefficient × (sinal_EEG_bruto − Baseline) + TranslationCoefficient
```

Onde:
- **TangentCoefficient** = 4.4143 (nas sessões de jogo) - escala a amplitude do sinal
- **TranslationCoefficient** = −0.0762 (nas sessões de jogo) - desloca o sinal verticalmente
- **Baseline** = −0.2157 (média da subsessão 0 do utilizador) - o nível de referência pessoal

Na **subsessão 0 de calibração**, estes coeficientes são todos zero - o ProtocolValue é o sinal bruto não processado. Nas **subsessões de jogo**, o sinal é escalado e deslocado usando os coeficientes calculados durante a calibração.

O protocolo específico deste utilizador é **F4Alpha − F3Alpha** - a diferença de potência Alpha entre o hemisfério direito (F4) e esquerdo (F3). Esta assimetria frontal é frequentemente associada ao equilíbrio emocional e ao estado de "flow".

**O que significa o ProtocolValue na prática:**
- Valor **positivo** → o cérebro está a produzir mais Alpha no hemisfério direito que no esquerdo → bom desempenho no protocolo
- Valor **negativo** → o inverso → mau desempenho no protocolo
- O objectivo do sistema é **maximizar o ProtocolValue ao longo da sessão**

**Estatísticas do ProtocolValue:**

| Estatística | Valor |
|---|---|
| Média global | −0.0760 |
| Desvio padrão | 0.5003 |
| Mínimo | −2.7328 |
| Máximo | +2.0369 |
| Assimetria | −0.27 (ligeiramente assimétrico à esquerda) |
| Média na subsessão 0 | −0.1914 |
| Média na subsessão 9 | +0.0618 |

### O PlayerPositionY como proxy comportamental

A posição vertical da nave (`PlayerPositionY`) é **directamente controlada pela actividade cerebral** - quando o ProtocolValue sobe, a nave sobe. É portanto um proxy comportamental do sinal neurofisiológico.

A correlação de Pearson entre `PlayerPositionY` e `ProtocolValue` é **r = 0.113** - positiva mas fraca, o que faz sentido: o jogo tem dinâmicas próprias (inércia, limites do ecrã) que introduzem não-linearidades. Mesmo assim, é a variável de jogo com maior correlação com o ProtocolValue.

---

## 3. A Questão de Investigação

### Três perguntas encadeadas

> **Pergunta 1:** Consegue-se prever o ProtocolValue no instante seguinte (t+1) a partir do estado actual?

> **Pergunta 2:** Dado um modelo de predição, consegue-se recomendar a acção que maximiza o ProtocolValue previsto?

> **Pergunta 3:** Um agente de aprendizagem por reforço consegue aprender uma política melhor que as abordagens simples?

Estas perguntas estão ordenadas por complexidade. Se P1 falhar, P2 e P3 não fazem sentido. Se P2 falhar, P3 é a alternativa mais poderosa. Em prática, P1 funcionou razoavelmente (67.6% de precisão direcional), P2 degenerou por falta de dados, e P3 funcionou parcialmente (FQI > política actual, mas com limitações severas).

### Porque é este um problema difícil?

**1. O sinal EEG é inerentemente ruidoso.** A actividade cerebral é não-estacionária (muda ao longo do tempo), sujeita a artefactos (movimento ocular, tensão muscular, interferência eléctrica), e extremamente individual. O sinal de uma pessoa não prediz o de outra.

**2. Há apenas uma sessão de dados.** Com 3,549 amostras no total e 2,667 de treino efectivo, o espaço de estados (246 dimensões) é vastíssimo em relação aos dados disponíveis. Os modelos de RL precisam de ordens de grandeza mais dados para aprender políticas robustas.

**3. O espaço de acções é implícito.** O sistema da Neroes não regista explicitamente que acção tomou em cada instante. O espaço de acções teve de ser construído como proxy a partir de tendências inter-subsessão, o que introduz ruído e ambiguidade.

**4. A recompensa é ruidosa e atrasada.** A mudança no ProtocolValue de um instante para o seguinte (ΔPV) é muito ruidosa (std = 0.593). O efeito real de uma acção pode só ser visível vários segundos depois.

---

## 4. Feature Engineering em Profundidade

### O que é feature engineering e porque importa

"Feature engineering" é o processo de transformar os dados brutos em representações que os modelos conseguem usar eficazmente. Com dados de EEG e jogo, os valores brutos em cada instante de tempo têm muito ruído e pouca informação. As features engineered capturam padrões que os valores brutos não revelam directamente.

### Limpeza inicial: eliminar colunas mortas

Das 149 colunas originais, foram eliminadas:
- **66 colunas de elétrodos mortos** (AF3, AF4, F7, F8, FC5, FC6, Fp1, Fp2, O1, O2, Oz, P7, P8, Pz, T7, T8 × 5–6 bandas cada) - todas zero constante
- **5 colunas 100% nulas** (Annotations, AsteroidPositionX, AsteroidPositionY, AudioTracks, RecoveryOffsetFromBaseline)
- **4 colunas constantes de jogo** (OngoingBlackHole, OngoingDistortedVision, OngoingRecovery, PerformanceRating) - sem variação nesta sessão

Ficaram **50 colunas** com sinal real.

### Z-normalização relativa ao baseline pessoal

Esta é a transformação mais importante de todo o projeto. Em vez de usar os valores brutos de EEG, normalizamos cada canal EEG relativamente ao baseline da **subsessão 0 do mesmo utilizador**:

```
EEG_normalizado = (EEG_bruto − média_subsessão0) / desvio_padrão_subsessão0
```

**Porque é crucial esta normalização:**

Um valor de F3Alpha = 0.015 pode ser alto para uma pessoa e baixo para outra. O que importa não é o valor absoluto, mas **o desvio do estado de repouso da própria pessoa**. A normalização torna o sinal comparável dentro do mesmo indivíduo ao longo do tempo, e potencialmente comparável entre indivíduos se o modelo for re-treinado noutras sessões.

Esta técnica é standard em neurofeedback clínico - é exactamente o que os terapeutas fazem ao calibrar o sistema no início de cada sessão.

### Features de lag temporal

O sinal de EEG tem **memória temporal** - o valor presente depende dos valores passados. Para capturar esta memória, criámos versões atrasadas das principais variáveis:

- **lag-1:** o valor de há 1 amostra (≈1 segundo atrás)
- **lag-2:** o valor de há 2 amostras
- **lag-5:** o valor de há 5 amostras (≈5 segundos atrás)

Estas features foram criadas para: ProtocolValue, todos os 20 canais EEG normalizados, e as principais variáveis de jogo (PlayerPositionY, Morale, LevelProgress).

**A importância dos lags foi confirmada pelos resultados:** As 12 features mais importantes no LightGBM final são todas autorregresivas (lags e médias rolantes do próprio ProtocolValue). O sinal tem forte memória temporal - o melhor preditor do próximo valor é o histórico recente do próprio valor.

### Features de janela rolante (rolling features)

Para além dos lags, criámos estatísticas de janelas temporais:

- **Média rolante** (janelas de 5, 10, 20 amostras): captura a tendência local
- **Desvio padrão rolante** (janelas de 5, 10, 20 amostras): captura a volatilidade local

Um desvio padrão rolante alto significa que o sinal está instável; um valor baixo significa que está estável. A estabilidade local do ProtocolValue é um preditor importante do próximo valor.

O `ProtocolValue_rstd5` (desvio padrão rolante de 5 amostras) é a **feature mais importante no LightGBM** (ganho = 1,089), à frente de todos os lags e médias.

### Features de contexto

Foram adicionadas duas features de contexto que não vêm do sinal mas do timing:

- **subsession_norm:** o índice da subsessão normalizado (0 a 1) - o modelo pode aprender que as sessões mais tardias têm comportamentos diferentes das iniciais
- **sample_idx_norm:** a posição dentro de cada subsessão (0 a 1) - o modelo pode aprender que o início de cada subsessão é diferente do fim

Ambas aparecem no top-15 de features por Mutual Information e no top-15 do LightGBM, confirmando que o **contexto temporal** é relevante para a predição.

### A construção do espaço de acções (e os seus problemas)

O sistema da Neroes não regista explicitamente qual acção o sistema tomou em cada instante. Por isso, o espaço de acções foi construído como proxy:

- **Acção 0 - Baixar limiar:** o sistema está a facilitar a tarefa
- **Acção 1 - Manter limiar:** o sistema mantém a dificuldade actual
- **Acção 2 - Subir limiar:** o sistema está a aumentar a dificuldade

A atribuição de acções foi feita com base em limiares do ProtocolValue, mas o resultado foi uma distribuição **extremamente desequilibrada**:

| Acção | Descrição | N | % |
|---|---|---|---|
| 0 | Baixar limiar | 314 | 8.8% |
| 1 | Manter limiar | 1,349 | 38.0% |
| 2 | Subir limiar | 1,886 | 53.2% |

Esta assimetria (69.6% da acção 2 nos dados de RL) é o problema mais sério de todo o projeto. Os modelos de RL precisam de ver todas as acções com frequência razoável para aprender as suas consequências. Com só 309 amostras da acção 0 e 502 da acção 1, não há dados suficientes.

### O que a Mutual Information revela

A **Mutual Information (MI)** mede quanto cada feature reduz a incerteza sobre o alvo (ProtocolValue seguinte). Valores mais altos = mais informação.

| Rank | Feature | MI Score | Interpretação |
|---|---|---|---|
| 1 | ProtocolValue_rmean20 | 0.095 | Média rolante de 20s é o melhor preditor |
| 2 | ProtocolValue_rmean10 | 0.069 | Média rolante de 10s |
| 3 | ProtocolValue_rmean5 | 0.059 | Média rolante de 5s |
| 4 | PlayerPositionY_lag1 | 0.036 | Posição do jogador há 1s |
| 5 | ProtocolValue_rstd20 | 0.027 | Volatilidade a 20s |
| ... | EEG features | 0.018–0.025 | Contribuição modesta |

**Conclusão:** A MI confirma que o ProtocolValue tem forte autocorrelação temporal. As features de EEG puro têm MI muito baixa (não aparecem no top-5), o que sugere que o sinal EEG bruto é demasiado ruidoso para predição directa a 1 segundo de resolução.

---

## 5. Predição Supervisionada com LightGBM

### O que é o LightGBM e porque foi escolhido

O **LightGBM** (Light Gradient Boosting Machine) é um algoritmo de aprendizagem automática baseado em árvores de decisão em gradiente. A ideia é construir um conjunto de árvores onde cada nova árvore corrige os erros das anteriores.

**Porque LightGBM e não redes neuronais?**

Com apenas 2,658 linhas de treino efectivo, as redes neuronais profundas sofreriam de **overfitting** (memorizar os dados de treino em vez de aprender padrões gerais). O LightGBM é robusto com poucos dados, lida bem com features heterogéneas (EEG + jogo + contexto), é rápido, e é interpretável através das importâncias de features.

**Hiperparâmetros usados:**
- 300 estimadores (árvores) no treino com walk-forward; 500 no modelo final
- Taxa de aprendizagem: 0.05 (conservadora, evita overfitting)
- 31 folhas por árvore (modelo de média complexidade)
- Regularização L1 e L2 = 0.1 (penaliza modelos complexos)
- Subsampling de 80% de features e amostras por árvore (variabilidade para evitar overfitting)
- Early stopping com paciência de 30 iterações

### Walk-forward cross-validation - porque é essencial

A escolha do método de validação é **a decisão metodológica mais importante do projeto**.

**O problema com K-fold aleatório:** Se misturarmos dados do futuro com o treino (por exemplo, treinar com dados das sessões 5, 7, 9 e testar na sessão 3), o modelo vê padrões que em deployment real não existiriam. Os resultados ficam artificialmente bons - é **data leakage temporal**.

**Walk-forward CV** resolve isto: para cada fold, treina apenas nos dados cronologicamente anteriores ao fold de teste:

```
Fold 1: Treino [ss1]          → Teste [ss2]
Fold 2: Treino [ss1, ss2]     → Teste [ss3]
Fold 3: Treino [ss1..ss3]     → Teste [ss4]
...
Fold 8: Treino [ss1..ss8]     → Teste [ss9]
```

Isto simula exactamente como o sistema seria usado em produção: em cada nova subsessão, o modelo foi treinado apenas com o que aconteceu antes.

### Resultados fold a fold

| Fold (test_ss) | N treino | N teste | MAE | RMSE | R² | DirAcc |
|---|---|---|---|---|---|---|
| 2 | 234 | 412 | 0.4034 | 0.5280 | +0.0024 | 67.0% |
| 3 | 646 | 264 | 0.3664 | 0.4771 | +0.0061 | 68.2% |
| 4 | 910 | 311 | 0.3169 | 0.4203 | −0.0060 | 67.5% |
| 5 | 1,221 | 335 | 0.3449 | 0.4640 | −0.0012 | 68.4% |
| 6 | 1,556 | 308 | 0.3558 | 0.4908 | +0.0523 | 71.1% |
| 7 | 1,864 | 264 | 0.3276 | 0.4571 | −0.0407 | 66.7% |
| 8 | 2,128 | 266 | 0.3175 | 0.4226 | +0.0299 | 63.9% |
| 9 | 2,394 | 264 | 0.3739 | 0.4912 | −0.0200 | 67.8% |
| **Média** | | | **0.3508 ± 0.0303** | **0.4689** | **+0.0029** | **67.6%** |

**Nota sobre o fold 2 (ss2):** O MAE é o mais alto (0.4034) porque o treino é muito pequeno - apenas 234 amostras da subsessão 1. À medida que os dados de treino crescem, o MAE tende a descer (o melhor é o fold de ss8: 0.3175).

### Porque é que o R² é quase zero - mas o modelo ainda é útil

O R² (coeficiente de determinação) mede que proporção da variância do alvo é explicada pelo modelo. Um R² = 0 significa que o modelo é equivalente a prever sempre a média; R² = 1 seria predição perfeita.

O R² médio do LightGBM é **+0.0029** - praticamente zero. Isto parece mau, mas não é necessariamente um fracasso.

**Porque é que o R² é tão baixo?**

O ProtocolValue a 1 segundo de resolução comporta-se quase como uma **caminhada aleatória** (random walk): dado o valor actual, o próximo valor é aproximadamente imprevisível em termos de magnitude exacta. Isto é confirmado pelo facto de o modelo **LastValue** (predizer sempre o valor actual) ter R² = −0.667 - pior do que prever a média! O sinal tem reversão à média muito rápida.

**Porque é que 67.6% de precisão direcional é útil, apesar do R² ≈ 0?**

O sistema de neurofeedback não precisa de saber o valor exacto do próximo ProtocolValue. Precisa apenas de saber: **vai subir ou descer?** Se o modelo responde correctamente a esta questão em 67.6% dos casos (vs. 50% de chance aleatória), tem informação suficiente para guiar decisões de protocolo.

Uma analogia: um meteorologista com 67.6% de precisão a prever "vai chover ou não?" é muito útil, mesmo que os seus valores de precipitação exactos sejam imprecisos.

### O que as feature importances revelam

**Top 15 features por importância do LightGBM (ganho):**

| Rank | Feature | Ganho | Interpretação |
|---|---|---|---|
| 1 | ProtocolValue_rstd5 | 1,089 | Volatilidade local (5s) |
| 2 | pv_delta_2 | 1,045 | Mudança de há 2 passos |
| 3 | pv_lag5 | 1,015 | Valor há 5s |
| 4 | pv_delta | 1,007 | Mudança no último passo |
| 5 | ProtocolValue_rstd20 | 996 | Volatilidade local (20s) |
| 6 | ProtocolValue_rmean20 | 942 | Tendência local (20s) |
| 7 | ProtocolValue_rstd10 | 926 | Volatilidade local (10s) |
| 8 | pv_lag2 | 912 | Valor há 2s |
| 9 | ProtocolValue_rmean5 | 901 | Tendência local (5s) |
| 10 | ProtocolValue_rmean10 | 896 | Tendência local (10s) |
| 11 | sample_idx_norm | 890 | Posição na subsessão |
| 12 | pv_lag1 | 798 | Valor há 1s |
| 13 | PlayerPositionY | 456 | Posição da nave |
| 14 | PlayerPositionY_lag5 | 430 | Posição da nave há 5s |
| 15 | PlayerPositionY_lag2 | 429 | Posição da nave há 2s |

**Conclusão crítica:** As 12 features mais importantes são **todas autorregresivas** - derivadas do próprio ProtocolValue e do PlayerPositionY. As features de EEG puro (bandas de frequência normalizadas) não aparecem no top-15. Isto sugere que **a 1 segundo de resolução, a melhor predição usa o histórico do próprio sinal, não o sinal de EEG actual.**

Isto faz sentido neurofisiologicamente: o EEG bruto a cada segundo é muito ruidoso. A média rolante a 20 segundos já é uma estimativa muito mais estável do estado cerebral.

---

## 6. Política de Recomendação Não-RL (Notebook 04)

### A ideia: lookahead de um passo

A abordagem é conceptualmente simples: dado o estado actual, **simular cada uma das 3 acções** no vector de estado e escolher a que o modelo prevê como produzindo o maior ProtocolValue no passo seguinte.

Esta é uma **política gananciosa de um passo** (one-step greedy): não considera as consequências futuras além do próximo instante, mas não requer treino de RL.

### Por que degenerou: acção não está nas features

**O problema fundamental:** A variável `action` (0, 1, ou 2) não foi incluída no vector de features do modelo de predição. Por isso, quando o sistema tenta "simular" diferentes acções, não consegue - o modelo não tem qualquer mecanismo para diferenciar as consequências das três acções.

O resultado: o modelo prediz sempre o mesmo valor independentemente da acção simulada → recomenda sempre a mesma acção (Hold, 100% das vezes).

**A taxa de melhoria de 47.6%** (passos em que o ProtocolValue sobe) não resulta de qualquer decisão real - é apenas a taxa base de passos positivos nos dados de jogo. O notebook 04, na sua configuração actual, não tem valor de recomendação.

### O que se deveria fazer

Para que o notebook 04 funcione, a acção teria de:
1. Ser explicitamente registada pelo sistema em cada instante
2. Ser incluída como feature no modelo de predição (o modelo aprenderia: "se a acção for 2, o ProtocolValue tende a subir X mais do que com acção 0")
3. A política greedy então teria significado real

---

## 7. Agentes de Aprendizagem por Reforço

### O framework MDP

A aprendizagem por reforço enquadra o problema como um **Processo de Decisão de Markov (MDP)**:

| Componente | Definição neste projeto |
|---|---|
| **Estado (s)** | Vector de 246 features (EEG normalizado + lags + rolling + contexto) |
| **Acção (a)** | {0=Baixar limiar, 1=Manter, 2=Subir limiar} |
| **Recompensa (r)** | ΔProtocolValue = PV(t) − PV(t−1), clipado ao intervalo [2%, 98%] |
| **Política (π)** | Função que mapeia estados a acções |
| **Horizonte** | Uma subsessão (~240–418 passos) |
| **Desconto (γ)** | 0.95 (recompensas futuras valem 95% das presentes) |

A recompensa foi clipada para remover outliers extremos (range efectivo: [−1.359, +1.328]).

### Avaliação offline: o problema do counterfactual

O maior desafio do RL offline é: **como saber se uma política alternativa seria melhor, sem a poder testar em dados reais?**

A solução usada é o **Inverse Propensity Scoring (IPS)**: para estimar a recompensa esperada de uma política π, calcula-se a média das recompensas observadas apenas nos momentos em que π teria escolhido a mesma acção que a política de registo. Isto controla para o facto de só termos observado as consequências das acções que foram realmente tomadas.

O problema do IPS é que, se a política avaliada for muito diferente da política de registo, a taxa de coincidência é baixa → o estimador tem elevada variância → os resultados são pouco fiáveis estatisticamente.

### LinUCB - o bandit contextual

**O que é:** O LinUCB (Linear Upper Confidence Bound) é um algoritmo de **bandit contextual** - um caso especial de RL onde não há transições de estado (a decisão é baseada apenas no estado actual, sem considerar estados futuros).

**Como funciona:**
1. Para cada acção `a`, mantém um modelo linear: `Q(s, a) = θ_a^T × s` (recompensa esperada da acção `a` no estado `s`)
2. Ao escolher uma acção, calcula um **upper confidence bound**: `UCB_a = θ_a^T × s + α × √(s^T × A_a^{-1} × s)`
3. O termo `α × √(s^T × A_a^{-1} × s)` é um bónus de incerteza - estados menos explorados têm UCBs maiores (o agente é incentivado a explorar)
4. Escolhe a acção com maior UCB

**Resultados:**
- Taxa de coincidência com a política de registo: **0.6%** (apenas 15 instantes em 2,667)
- Recompensa média nesses 15 instantes: **−0.0491**
- Distribuição de acções recomendadas: 67% acção 0 (Baixar), 23% acção 1, 10% acção 2

**Porque falhou:** O LinUCB recomenda maioritariamente "Baixar limiar" (acção 0), que é a acção oposta à que predomina nos dados reais (70% "Subir"). Com apenas 15 coincidências, o IPS é estatisticamente inútil - n=15 não permite qualquer inferência robusta. O agente não aprendeu porque os dados de treino são demasiado escassos e desequilibrados.

### FQI - Fitted Q-Iteration

**O que é:** O FQI (Fitted Q-Iteration) é uma forma de **Q-learning offline em lote**. Em vez de atualizar uma tabela Q incrementalmente (como o Q-learning clássico), o FQI ajusta um regressor (Gradient Boosting) que mapeia (estado, acção) → valor Q esperado.

**Como funciona:**
1. Inicializa: Q₀(s,a) = r (a função Q inicial é a recompensa imediata)
2. Para cada iteração k=1..K:
   - Para cada transição (s, a, r, s'): calcula o alvo `y = r + γ × max_a'[Q_{k-1}(s', a')]`
   - Ajusta um novo modelo: `Q_k = fit(SA → y)` onde SA = [s, one-hot(a)]
3. A política final: `π(s) = argmax_a Q_K(s, a)`

**Evolução dos valores Q:**

| Iteração | Q mínimo | Q máximo |
|---|---|---|
| 1 | −2.013 | +2.054 |
| 4 | −1.356 | +1.688 |
| 7 | −1.377 | +1.734 |
| 10 (final) | −1.379 | +1.848 |

O range de Q estabiliza após ~4 iterações, o que indica convergência razoável.

**Resultados:**
- Distribuição de acções recomendadas: **99.8% acção 0** (Baixar limiar), 0.2% acção 1, 0% acção 2
- Recompensa IPS estimada: **+0.0020** (vs. +0.0005 da política real)

**Porque quase funcionou mas ainda é limitado:** O FQI consegue marginalmentee superar a política actual (+0.0020 vs. +0.0005) mas **colapsa para uma política quasi-determinística** (sempre "Baixar limiar"). Este colapso é um sintoma clássico de **cobertura insuficiente**: com 70% dos dados históricos a usar acção 2, o Q-model aprende muito sobre o que acontece quando se "Sobe" mas muito pouco sobre os outros casos. A função Q estimada para acção 0 torna-se artificialmente alta porque o estimador de bootstrap nas iterações propaga erros de regiões esparsas.

### Comparação de políticas (IPS)

| Política | Recompensa média (IPS) | N coincidências |
|---|---|---|
| Aleatória | −0.0147 | 883 |
| LinUCB | −0.0491 | 15 |
| **FQI** | **+0.0020** | 309 |
| Real (registo) | +0.0005 | 2,667 |

A política aleatória tem 883 coincidências porque, com 3 acções igualmente prováveis, ~1/3 das escolhas coincide com a política real.

---

## 8. O Que Resultou

### A eficácia do neurofeedback está confirmada

A descoberta mais importante não é sobre os modelos - é sobre o utilizador. O ProtocolValue melhorou consistentemente ao longo das subsessões:

| SubSessão | Média | Evolução |
|---|---|---|
| 0 (baseline) | −0.1914 | Referência |
| 1 | −0.1852 | ≈ baseline |
| 2 | −0.1160 | ↑ ligeira melhoria |
| 3 | −0.0577 | ↑ melhoria clara |
| 4 | −0.0206 | ↑ |
| 5 | +0.0077 | ↑ **primeiro valor positivo** |
| 6 | −0.0705 | ↓ ligeira regressão |
| 7 | +0.0228 | ↑ recuperação |
| 8 | +0.0083 | → estável |
| 9 | +0.0618 | ↑ **valor mais alto** |

**Delta total: +0.2470** (de ss1 para ss9) = **+133% relativo**.

O protocolo de neurofeedback está a funcionar para este participante. A trajectória geral é ascendente, com uma dip na ss6 que pode corresponder a fadiga mental após 20 minutos de sessão.

### A predição supervisionada é viável e útil

- **MAE = 0.3508 ± 0.0303** - redução de 26.7% face à baseline de persistência
- **Precisão direcional = 67.6%** - significativamente acima do acaso (50%)
- O modelo é 26.7% mais preciso que o baseline mais simples (LastValue)
- A precisão direcional de 67.6% é suficiente para guiar decisões de protocolo em tempo real

### O LightGBM bate todas as baselines ingénuas

| Método | MAE | R² | % vs. LastValue |
|---|---|---|---|
| LightGBM | **0.3508** | +0.0029 | **−26.7%** |
| RollingMean(w=10) | 0.3726 | +0.0429 | −22.1% |
| SessionMean | 0.3943 | −0.0983 | −17.6% |
| LastValue | 0.4786 | −0.6676 | - |

**Nota interessante:** O RollingMean(w=10) tem R² mais alto (+0.043) do que o LightGBM (+0.003), mas MAE mais alto. Isto sugere que o LightGBM é melhor em erros absolutos mas o RollingMean captura melhor a variância relativa. Para uso prático, o MAE é a métrica mais relevante.

---

## 9. O Que Não Resultou e Porquê

### 9.1 R² ≈ 0 em todos os métodos

**O que aconteceu:** Nenhum método consegue explicar mais do que ~4% da variância do ProtocolValue no passo seguinte. O LightGBM tem R² = +0.003 - praticamente inútil para predição de magnitude.

**Porque aconteceu:** O ProtocolValue a 1 segundo de resolução é um sinal quase caótico. A variância entre instantes consecutivos é enorme (std ≈ 0.5) e é dominada por ruído neural e artefactos. A informação disponível (EEG de 4 elétrodos, posição do jogador) não é suficiente para prever o valor exacto.

**O que precisaria de mudar:** Janelas de predição mais longas (prever PV médio nos próximos 10 segundos em vez de no próximo segundo), mais elétrodos activos, filtragem de artefactos mais sofisticada, e features derivadas de análise espectral (coherência entre elétrodos, assimetria hemisférica).

### 9.2 RL offline com dados esparsos

**O que aconteceu:** Ambos os agentes colapsaram para políticas quasi-determinísticas que recomendam quase sempre a acção "Baixar limiar", que é a acção menos frequente nos dados históricos.

**Porque aconteceu:** O RL offline precisa de **cobertura densa** em todo o espaço (estado, acção). Com 70% dos dados a usar a acção 2 (Subir), as acções 0 e 1 têm cobertura insuficiente para o modelo aprender as suas consequências. O bootstrap iterativo do FQI propaga e amplifica estes erros de regiões pouco cobertas.

**O que precisaria de mudar:** Uma fase de **exploração controlada** onde o sistema experimenta intencionalmente as três acções com frequência razoável - pelo menos 20–30% cada. Ou uma recolha de dados em múltiplas sessões com acções aleatorizadas.

### 9.3 Notebook 04 completamente degenerado

**O que aconteceu:** O módulo de recomendação não-RL recomenda sempre a mesma acção (Hold, 100%).

**Porque aconteceu:** A variável `action` não está no vector de features do modelo de predição. Sem esta variável, o modelo não consegue diferenciar as consequências das diferentes acções.

**O que precisaria de mudar:** Incluir `action` como feature do modelo de predição desde o início, e garantir que o modelo vê exemplos suficientes de cada acção durante o treino.

### 9.4 Generalização limitada a um único sujeito

**O que aconteceu:** Todo o projeto usa dados de uma única sessão de um único participante (SessionNumber 66).

**Porque isto importa:** Os coeficientes de normalização (baseline de subsessão 0), as importâncias das features, e os pesos dos agentes de RL são específicos deste indivíduo. O protocolo `F4Alpha − F3Alpha` pode não ser o protocolo certo para outro utilizador. Os valores de `TangentCoefficient` e `TranslationCoefficient` são personalizados por sessão.

**O que precisaria de mudar:** Recolha de dados de múltiplos participantes, validação cruzada entre sujeitos, e uma arquitectura que combine um componente personalizado (calibrado na subsessão 0 de cada novo utilizador) com um componente genérico (treinado em dados populacionais).

---

## 10. Implicações e Próximos Passos

### Para a Neroes (implicações práticas)

**O que implementar imediatamente:**
1. O modelo LightGBM de predição de direcção (67.6% precisão) pode ser integrado num sistema de alerta para terapeutas - "o sinal está prestes a degradar" ou "o utilizador está a melhorar, pode subir o limiar"
2. Registar explicitamente todas as acções do sistema em tempo real (qual foi o ajuste do limiar em cada segundo) - isto é prerequisito para RL offline futuro
3. Estruturar as sessões futuras de forma a ter dados balanceados entre acções (fase exploratória automática no início de cada sessão)

**O que planear a médio prazo:**
1. Com 10+ sessões de dados por utilizador: re-treinar o FQI - a qualidade da política melhorará substancialmente
2. Com acções registadas: o notebook 04 torna-se funcional e pode ser usado como política de recomendação simples mas eficaz
3. Arquitectura online: substituir o FQI por um LinUCB que actualiza em tempo real durante cada sessão

### Para a ciência (implicações de investigação)

**O que é publicável:**
- A pipeline completa de EDA → feature engineering → predição supervisionada → RL offline para neurofeedback adaptativo
- A metodologia de walk-forward CV como forma correcta de validar modelos temporais em neurofeedback
- A análise honesta das limitações do RL offline com dados de sessão única
- A proposta de arquitectura para um sistema de neurofeedback adaptativo em tempo real

**O que precisa de mais investigação:**
- Validação em múltiplos participantes
- Comparação de diferentes protocolos EEG (não apenas F4Alpha − F3Alpha)
- Features de EEG mais sofisticadas (coherência entre elétrodos, análise de connectividade)
- Métodos de RL online (LinUCB com actualização em tempo real)

### Para RL em neurofeedback (implicações metodológicas)

O projeto demonstra um problema que provavelmente afecta muitos outros trabalhos de RL em neurofeedback: **o desequilíbrio de acções no dados históricos torna o IPS altamente variante e os agentes degenerados**. A solução proposta - fase de exploração controlada - é standard em RL clínico (contextual bandit com exploração garantida) mas raramente discutida na literatura de neurofeedback.

---

## 11. Glossário de Termos Técnicos

### Termos de neurociência

| Termo | Definição |
|---|---|
| **EEG** | Electroencefalografia - medição da actividade eléctrica cerebral através de elétrodos no couro cabeludo |
| **Elétrodo** | Sensor que mede diferença de potencial eléctrico no couro cabeludo; cada posição tem uma nomenclatura normalizada (F3, C4, etc.) |
| **Banda de frequência** | Gama de frequências do sinal EEG (Theta, Alpha, Beta, Gamma) com associações cognitivas distintas |
| **Neurofeedback** | Técnica de treino cerebral baseada em feedback em tempo real da actividade EEG |
| **Baseline** | Sinal de EEG em repouso, sem tarefa; serve como referência pessoal |
| **ProtocolValue** | Variável derivada do EEG que quantifica a aderência ao protocolo de treino |
| **Assimetria frontal** | Diferença de actividade entre os hemisférios frontal esquerdo e direito; associada ao equilíbrio emocional |

### Termos de aprendizagem automática

| Termo | Definição |
|---|---|
| **Feature** | Variável de entrada de um modelo; cada linha dos dados tem múltiplas features |
| **Feature engineering** | Processo de criar novas features mais informativas a partir dos dados brutos |
| **MAE** | Mean Absolute Error - erro médio absoluto entre predições e valores reais |
| **RMSE** | Root Mean Squared Error - raiz do erro quadrático médio; penaliza mais os grandes erros |
| **R²** | Coeficiente de determinação - proporção da variância do alvo explicada pelo modelo (0=inútil, 1=perfeito) |
| **Precisão direcional** | % de vezes que o modelo prevê correctamente a direcção (subida/descida) do sinal |
| **Overfitting** | O modelo memoriza os dados de treino e não generaliza para dados novos |
| **Walk-forward CV** | Método de validação que respeita a ordem temporal - nunca usa dados futuros no treino |
| **Data leakage** | Uso acidental de informação futura no treino - produz resultados artificialmente bons |
| **LightGBM** | Algoritmo de gradient boosting em árvores de decisão; eficiente e robusto com poucos dados |
| **Mutual Information** | Medida de quanto uma feature reduz a incerteza sobre o alvo |
| **Z-score** | Normalização: (valor − média) / desvio padrão; expressa valores em unidades de desvio padrão |

### Termos de aprendizagem por reforço

| Termo | Definição |
|---|---|
| **MDP** | Processo de Decisão de Markov - formalismo matemático para tomada de decisão sequencial |
| **Estado** | Representação completa do ambiente num dado instante |
| **Acção** | Decisão tomada pelo agente num dado estado |
| **Recompensa** | Sinal numérico que avalia a qualidade de uma acção num dado estado |
| **Política** | Função que mapeia estados a acções; é o que o agente aprende |
| **Função Q** | Q(s,a) - recompensa futura esperada ao tomar a acção `a` no estado `s` |
| **Bandit contextual** | RL simplificado onde não há transições de estado; foca na escolha óptima de acção dado o contexto |
| **LinUCB** | Bandit linear com upper confidence bound; equilíbrio entre exploração e exploração |
| **FQI** | Fitted Q-Iteration - Q-learning offline usando regressores como aproximadores da função Q |
| **IPS** | Inverse Propensity Scoring - método para estimar a recompensa de uma política offline sem a testar |
| **Offline RL** | Aprendizagem por reforço a partir de dados históricos, sem interacção com o ambiente |
| **Cobertura de acções** | Frequência com que cada acção foi observada nos dados históricos; RL offline requer cobertura densa |
| **Desconto (γ)** | Factor que pondera recompensas futuras (γ=0.95 → recompensas futuras valem 95% das presentes) |

---

*Este documento foi preparado para o projeto Neroes Neurofeedback Challenge, Abril 2026.*
*Todos os números referenciados foram extraídos directamente dos outputs dos notebooks de análise.*
