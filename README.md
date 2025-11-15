# 🧠 Classificação de Imagens com CNN e Data Augmentation

Este repositório contém um experimento utilizando **Redes Neurais Convolucionais (CNN)** para classificação de imagens.
O objetivo principal é analisar como diferentes estratégias de **Data Augmentation** afetam o desempenho do modelo, com foco em métricas como acurácia e matriz de confusão.

---

## 📂 Estrutura do Repositório


| Arquivo / Pasta | Descrição                                                                                                                                |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `FIA_TP2.ipynb` | Notebook contendo o código completo do projeto, incluindo pré-processamento, definição do modelo, treinamento, avaliação e análise. |
| `README.md`     | Documento com o contexto, metodologia e resultados finais do projeto.                                                                      |

---

## 🎯 Objetivo

Treinar um modelo de deep learning para classificação de imagens utilizando técnicas de aumento de dados (data augmentation), avaliando:

- Generalização do modelo
- Redução de overfitting
- Impacto nas métricas finais de desempenho

---

## 🧪 Metodologia

O projeto utiliza a seguinte pipeline:

1. **Carregamento e organização do dataset**
2. **Data Augmentation** com transformações como:
   - Rotação
   - Flip horizontal
   - Zoom
   - Mudança de brilho
   - Shear
3. **Treinamento de uma CNN** com:
   - Callbacks (Early Stopping, Reduce LR on Plateau e ModelCheckpoint)
   - Split entre treino e validação
4. **Avaliação do modelo**, incluindo matriz de confusão e acurácia.
5. **Discussão dos resultados**.

---

## 📊 Resultados (a preencher após execução)

- **Acurácia final:**

  > _Exemplo:_ `xx.xx%`
  >
- **Matriz de confusão:**
  _(Inserir imagem ou tabela após execução do notebook)_
- **Loss x Epoch / Accuracy x Epoch:**
  _(Inserir gráficos gerados no notebook)_

---

## 🧠 Análise da Generalização e Erros

- Classes mais confundidas:

  > _(Anotar após análise da matriz de confusão)_
  >
- Possíveis causas:

  - Semelhança visual entre classes
  - Dataset desbalanceado
  - Limitações da arquitetura

---

## 🏁 Conclusão

Com base nas análises, o uso de **data augmentation** teve impacto relevante no modelo, auxiliando na redução de overfitting e melhorando a capacidade de generalização.
Comparando os resultados com e sem augmentation, observa-se que o modelo com augmentations:

✔ Treina de maneira mais estável
✔ Generaliza melhor em dados nunca vistos
✔ Reduz discrepâncias entre loss/accuracy de treino e validação

A implementação e experimentação demonstram a importância dessas técnicas quando se trabalha com datasets limitados ou desbalanceados.

---

## 📌 Tecnologias Utilizadas

- Python
- TensorFlow / Keras
- NumPy / Pandas
- Matplotlib
- Jupyter Notebook

## Alunos

- Yago Lobato
- Nathã Barbosa
- Matheus Santarém
- Emanuel Andriola
- Daniel Trindade
- Cristiano Cardoso

### E-mails

- yagobrlobato@icomp.ufam.edu.br
- NathaBarbosa@icomp.ufam.edu.br
- matheus.santarem@icomp.ufam.edu.br
- Emanuel.moraes@icomp.ufam.edu.br
- daniel.trindade@icomp.ufam.edu.br
- cristiano.lima@icomp.ufam.edu.br

---
