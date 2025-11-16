# 🧠 Classificação de Resíduos com CNN e Data Augmentation

Este repositório apresenta um experimento da disciplina **Fundamentos de Inteligência Artificial (FIA)** utilizando **Redes Neurais Convolucionais (CNN)** para classificar imagens de resíduos em 6 categorias:

`cardboard, glass, metal, paper, plastic, trash`.

Além de treinar um modelo de deep learning, o foco é **analisar o impacto do data augmentation** na capacidade de generalização do modelo, avaliando acurácia, matriz de confusão e comportamento das curvas de treino/validação.

---

## 📂 Estrutura do Repositório

| Arquivo / Pasta | Descrição                                                                                                   |
| --------------- | ----------------------------------------------------------------------------------------------------------- |
| `FIA_TP2.ipynb` | Notebook principal com carregamento do dataset, pré-processamento, definição do modelo, treinamento e análise. |
| `README.md`     | Documento com o contexto, metodologia, resultados e discussão final do projeto.                             |

---

## 🌍 Contexto do Problema

A triagem correta de resíduos é um desafio importante em ambientes urbanos e industriais. Separar adequadamente **papel, plástico, vidro, metal, papelão e lixo misto** reduz a contaminação dos recicláveis e aumenta a eficiência de cooperativas e sistemas automatizados.

Neste trabalho, tratamos a triagem como um problema de **classificação de imagens multiclasse**: dada a foto de um resíduo, o modelo deve decidir a qual das 6 classes (`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`) ela pertence.

O objetivo é conectar conceitos de IA e visão computacional com uma aplicação prática de sustentabilidade, explorando o uso de CNNs e data augmentation para melhorar o desempenho em um dataset relativamente pequeno e desbalanceado.

---

## 🎯 Objetivo

Treinar uma rede neural convolucional capaz de classificar imagens de resíduos em 6 classes e:

- Investigar o **efeito do data augmentation** no desempenho;
- Avaliar **overfitting x generalização** por meio das curvas de treino/validação;
- Analisar, via **matriz de confusão**, quais classes são mais confundidas;
- Discutir limitações do modelo e possíveis **trabalhos futuros**.

---

## 🗃️ Dataset

Foi utilizado um dataset público do Kaggle derivado do **TrashNet**, contendo imagens em 6 classes:

- `cardboard`
- `glass`
- `metal`
- `paper`
- `plastic`
- `trash`

Principais características:

- Aproximadamente alguns milhares de imagens no total;
- Distribuição **desbalanceada** entre as classes (por exemplo, `trash` tem bem menos exemplos que `paper` e `plastic`);
- Imagens RGB redimensionadas para **224×224** pixels;
- Split feito via `ImageDataGenerator` com `validation_split`, resultando em algo próximo de **85% treino / 15% validação**.

No notebook, há uma célula que contabiliza a quantidade de imagens em cada pasta e exibe alguns exemplos, evidenciando tanto a **variabilidade visual** quanto o **desbalanceamento** do conjunto de dados.

---

## 🧪 Metodologia

### 1. Pré-processamento e Data Augmentation

As imagens são carregadas com `ImageDataGenerator` em duas configurações:

- **Treino (`train_data`)**
  - `rescale = 1./255`
  - `validation_split` para separar treino/validação
  - Data augmentation **moderado**:
    - `rotation_range = 15`
    - `width_shift_range = 0.1`
    - `height_shift_range = 0.1`
    - `zoom_range = 0.1`
    - `horizontal_flip = True`
    - `fill_mode = 'nearest'`

- **Validação (`val_data`)**
  - Apenas `rescale = 1./255`
  - **Sem augmentation**, para garantir uma avaliação mais fiel ao comportamento real do modelo.

Durante o desenvolvimento também foram testadas configurações de augmentation **mais agressivas** (zoom alto, rotações grandes, flips verticais). Essas versões acabaram prejudicando a generalização, conforme discutido na seção de resultados.

---

### 2. Arquitetura da CNN

O modelo implementado segue a ideia proposta no enunciado (blocos convolucionais seguidos de pooling e um classificador denso), mas com capacidade ligeiramente aumentada em relação ao exemplo (mais filtros e uso de ZeroPadding2D):

- **Entrada:** imagem `224×224×3`, com um `ZeroPadding2D(padding=(1, 1))` para preservar melhor a dimensão espacial nas primeiras convoluções.
- **Blocos convolucionais + pooling:**
  - `Conv2D(64, 3×3, activation='relu')` → `Dropout(0.2)` → `MaxPool2D(2×2)`
  - `Conv2D(128, 3×3, activation='relu')` → `Dropout(0.2)` → `MaxPool2D(2×2)`
  - `Conv2D(128, 3×3, activation='relu')` → `Dropout(0.4)` → `MaxPool2D(2×2)`
- **Classificador totalmente conectado:**
  - `Flatten()`
  - `Dense(128, activation='relu')`
  - `Dropout(0.5)` para reduzir overfitting
  - `Dense(64, activation='relu')`
  - `Dense(6, activation='softmax')` (uma unidade para cada classe)

Configuração de treino:

- **Loss:** `CategoricalCrossentropy`
- **Métrica:** `CategoricalAccuracy`
- **Otimizador:** `Adam` com `learning_rate = 1e-4`

Callbacks:

- `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)`
- `ModelCheckpoint` para salvar o melhor modelo durante o treinamento.

---

### 3. Estratégia de Treinamento

- Execução realizada em **GPU (Google Colab)**;
- Número máximo de épocas em torno de **60–100**, mas na prática o `EarlyStopping` interrompe quando a validação deixa de melhorar;
- Avaliação final feita em cima do conjunto de validação, sem data augmentation, utilizando:
  - acurácia,
  - `classification_report`,
  - **matriz de confusão**.

---

## 📊 Resultados

Os números abaixo correspondem à melhor execução utilizando o notebook de referência (aquele que obteve a melhor combinação de acurácia e matriz de confusão):

- **Acurácia de treino:** ~**65%**
- **Acurácia de validação:** ~**~57%**  
  (248 acertos em 377 imagens no conjunto de validação)

### Desempenho por classe (validação)

A partir da matriz de confusão:

- `cardboard`: ≈ 62% de recall (37/60)
- `glass`: ≈ 52% de recall (39/75)
- `metal`: ≈ 74% de recall (45/61)
- `paper`: ≈ 87% de recall (77/89)
- `plastic`: ≈ 22% de recall (16/72)
- `trash`: ≈ 5% de recall (1/20)

O modelo tem desempenho muito bom em `paper` e `metal`, razoável em `cardboard` e `glass`, e encontra mais dificuldade em `plastic` e, principalmente, `trash`. A classe `trash` possui poucos exemplos e é muito heterogênea, o que explica o baixo recall e as confusões com `metal`, `paper` e `glass`.

---

### 📈 Curvas de Treino e Validação

![Gráfico de Treinamento](grafico_acuracia.png)

O gráfico de acurácia mostra que:

- As acurácias de treino e validação **aumentam ao longo das épocas**;
- O treino atinge cerca de **71%** e a validação estabiliza em torno de **60–66%**;
- Há um *gap* moderado entre as curvas (~0.06–0.1), indicando algum **overfitting leve**, esperado em datasets pequenos, mas **sem colapso**: a validação não despenca, o que indica boa capacidade de generalização.

---

### 🧮 Matriz de Confusão

![Matriz de Confusão](matriz_confusao.png)

A matriz de confusão evidencia que:

- `paper`, `metal`, `glass` e `cardboard` concentram grande parte dos acertos na diagonal;
- `plastic` é frequentemente confundido com `paper` e `glass` (por exemplo, garrafas plásticas x garrafas de vidro);
- `trash` é confundido com `metal`, `paper` e `plastic`, o que faz sentido por ser uma classe mais **genérica e heterogênea** e com menor número de exemplos.

---

---

## 🧠 Análise da Generalização e Erros

- Classes mais confundidas:

  > Lixo Orgânico (trash): Esta foi a classe com o desempenho mais baixo (recall de $\approx 40\%$). O modelo teve dificuldade em identificar corretamente os itens desta categoria.
  > Plástico (plastic): Foi a segunda classe mais difícil, com um recall de $\approx 57\%$.
  
- Possíveis causas:

  - Semelhança visual entre classes: A principal confusão provavelmente ocorre entre plastic e glass (vidro), já que ambos podem ser transparentes e ter formatos de garrafa, tornando-os difíceis de distinguir para a CNN.
  - Dataset desbalanceado: A classe trash é a menor de todo o dataset. Mesmo com o uso de class_weight (que ajudou muito), o modelo ainda teve menos exemplos para aprender os padrões dessa classe, o que explica seu baixo recall.
  - Limitações da arquitetura: Embora o modelo tenha atingido 71%, ele ainda é uma CNN personalizada. Classes muito complexas ou com poucas amostras podem exigir arquiteturas mais profundas (como as de Transfer Learning) para uma classificação perfeita.

---

## 🔁 Impacto do Data Augmentation

Durante o projeto foram testados três cenários principais:

1. **Sem data augmentation**  
   - O modelo aprendia rápido no conjunto de treino, mas a acurácia de validação ficava em torno de **0.40–0.45**;  
   - As curvas mostravam **overfitting**, com a acurácia de treino subindo bem mais que a de validação.

2. **Data augmentation muito agressivo**  
   - Foram aplicadas rotações muito grandes, zooms fortes e flips verticais;  
   - O modelo teve dificuldade para convergir e passou a confundir quase todas as classes, muitas vezes prevendo majoritariamente `plastic`;  
   - A acurácia de validação caiu para valores próximos de **0.25–0.30**, mostrando que transformações exageradas podem **distorcer demais o padrão visual** dos resíduos.

3. **Data augmentation moderado (configuração final)**  
   - Rotações menores, shifts e zoom leves e apenas flip horizontal;  
   - A acurácia de validação subiu para cerca de **0.60–0.66**;  
   - As curvas de treino/validação ficaram mais estáveis e a matriz de confusão passou a mostrar acertos bem distribuídos entre as classes.

**Conclusão:** o data augmentation é fundamental nesse problema, mas precisa ser **calibrado**. Com augmentations moderados, o modelo aprende melhor, reduz overfitting e generaliza bem para imagens nunca vistas.

---

## 🔍 Limitações

Limitações observadas:

- **Dataset pequeno e desbalanceado**, principalmente para a classe `trash`;
- **Semelhança visual** entre `plastic`, `glass` e `paper`, o que gera confusões frequentes;
- Arquitetura ainda relativamente simples se comparada a modelos de estado da arte.

---

## ⚙️ Tecnologias Utilizadas

- Python  
- TensorFlow / Keras  
- NumPy / Pandas  
- Matplotlib / Seaborn  
- Jupyter Notebook  
- Google Colab (GPU)

---

## 👥 Alunos

- Yago Lobato — yagobrlobato@icomp.ufam.edu.br  
- Nathã Barbosa — NathaBarbosa@icomp.ufam.edu.br  
- Matheus Santarém — matheus.santarem@icomp.ufam.edu.br  
- Emanuel Andriola — Emanuel.moraes@icomp.ufam.edu.br  
- Daniel Trindade — daniel.trindade@icomp.ufam.edu.br  
- Cristiano Cardoso — cristiano.lima@icomp.ufam.edu.br
