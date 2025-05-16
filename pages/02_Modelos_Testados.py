import streamlit as st

st.title("Modelos Testados")

st.header("1. Abordagem Inicial com Modelos Lineares")
st.markdown("""
Nesta primeira fase foram utilizados modelos lineares com vetorização textual e redução de dimensionalidade para estabelecer uma base comparativa. O pipeline incluiu:

- Vetorização textual (TF-IDF ou CountVectorizer)
- Redução via TruncatedSVD (100 componentes)
- Classificação com Regressão Logística
- Balanceamento com `class_weight='balanced'`

O objetivo era testar a capacidade de generalização com estruturas simples, avaliando os impactos de diferentes formas de vetorização.
""")

st.subheader("Modelos Baseline Testados")

st.markdown("""
**Modelo 1: TF-IDF (unigramas)**  
- Vetorizador: TF-IDF com ngram_range=(1,1) e max_features padrão  
- Redução de dimensionalidade: TruncatedSVD com 100 componentes  
- Variância explicada: 40.45%  
- F1-score (classe 1): 0.54  
- AUC: 0.669

**Modelo 2: CountVectorizer (1,2)**  
- Vetorizador: CountVectorizer com ngram_range=(1,2) e max_features=2000  
- Redução: TruncatedSVD com 100 componentes  
- Variância explicada: 71.52%  
- F1-score (classe 1): 0.54  
- AUC: 0.639

**Modelo 3: TF-IDF (1,2)**  
- Vetorizador: TF-IDF com ngram_range=(1,2) e max_features=3000  
- Redução: TruncatedSVD com 100 componentes  
- Variância explicada: 38.85%  
- F1-score (classe 1): 0.55  
- AUC: 0.674

> O modelo 3 apresentou os melhores resultados entre os lineares.
""")

st.header("2. Modelos Baseados em Árvores")

st.markdown("""
Na sequência, foram testados modelos mais robustos que lidam bem com interações entre variáveis e dados não linearmente separáveis:

**Random Forest**  
- n_estimators: 200  
- class_weight: balanced  
- AUC: 0.664  
- F1 Classe 1: 0.37  
- Recall: 0.27

**XGBoost (baseline)**  
- n_estimators: 200  
- max_depth: 6  
- learning_rate: 0.1  
- AUC: 0.663  
- F1 Classe 1: 0.47  
- Recall: 0.44

> O XGBoost teve desempenho superior à Random Forest na identificação da classe positiva.
""")

st.header("3. Otimização com RandomizedSearchCV")

st.markdown("""
O XGBoost foi então otimizado via RandomizedSearchCV, buscando melhores combinações de hiperparâmetros:

- Parâmetros testados: learning_rate, max_depth, n_estimators, subsample, colsample_bytree, gamma
- Objetivo: maximizar o F1-score da classe 1

**Melhor modelo encontrado:**  
- F1 Classe 1: 0.52  
- Recall: 0.57  
- AUC: 0.666

> A otimização trouxe um ganho consistente sem comprometer a robustez.
""")

st.header("4. Testes com Novas Colunas Textuais")

st.markdown("""
Para enriquecer a representação semântica dos candidatos, foram incorporadas novas colunas:
- Área de atuação profissional
- Conhecimentos técnicos
- Nível profissional

As colunas foram limpas, normalizadas e concatenadas em `texto_completo`. A vetorização TF-IDF (ngramas 1,2) foi reaplicada sobre essa nova coluna.

> Houve leve ganho de recall (de 57% para 59%) mantendo AUC e F1 estáveis.
""")

st.header("5. Stemming com RSLPStemmer")

st.markdown("""
Aplicou-se a técnica de stemming com RSLPStemmer (nltk) nas colunas textuais, reduzindo palavras à sua raiz morfológica. Essa técnica busca:
- Reduzir variação léxica
- Melhorar a generalização do vocabulário

**Resultados com stemming:**  
- F1 Classe 1: 0.54  
- Recall: 0.60  
- AUC: 0.671  
- Variância explicada: 40.26%

> O modelo mostrou ligeiros ganhos com essa abordagem.
""")

st.header("6. Sentence Embeddings com MiniLM")

st.markdown("""
Nesta etapa, foi adotada uma abordagem de vetorização textual com foco na representação semântica dos documentos. Utilizou-se o modelo `paraphrase-multilingual-MiniLM-L12-v2` da biblioteca `sentence-transformers`, capaz de gerar embeddings vetoriais densos e multilingues com alto poder expressivo. 

Essa abordagem tem o potencial de capturar o significado contextual das palavras e frases, indo além da frequência ou da coocorrência textual.

**Configuração do modelo:**
- Vetorização com embeddings MiniLM
- Classificação com XGBoost
- Otimização com GridSearchCV
- Balanceamento com `scale_pos_weight`

**Resultados:**
- F1 Classe 1: 0.47  
- Recall: 0.51  
- AUC: 0.618

> Apesar da sofisticação semântica, os resultados não superaram os modelos baseados em TF-IDF.
""")

st.header("7. Comparação Final de Modelos")

st.markdown("""
**Melhor modelo até o momento:**  
- Algoritmo: XGBoost otimizado com GridSearchCV  
- Pré-processamento: texto completo com stemming  
- Vetorização: TF-IDF (1,2)  
- Redução: TruncatedSVD (100 componentes)

**Métricas:**  
- F1 Classe 1: 0.54  
- Recall Classe 1: 0.60  
- AUC: 0.671

> O modelo final demonstrou maior equilíbrio e desempenho consistente mesmo frente a abordagens baseadas em embeddings densos.
""")
