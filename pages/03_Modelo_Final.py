import streamlit as st

st.title("Modelo Final")

st.header("1. Integração de Dados Candidato + Vaga")
st.markdown("""
Para o modelo final, foi realizada uma abordagem mais completa de integração de dados. O conjunto de dados combinou informações textuais do candidato (como o currículo e área de atuação) com descrições das vagas (como tipo de contratação, nível acadêmico e atividades previstas).

Essas colunas foram unificadas em um campo textual chamado `texto_completo`, que passou a ser a base de entrada para o modelo.
""")

st.header("2. Pipeline Final de Processamento")
st.markdown("""
A pipeline adotada seguiu as seguintes etapas:

- **Pré-processamento linguístico** com stemming utilizando o RSLPStemmer (nltk);
- **Vetorização textual** com TF-IDF, considerando n-gramas (1,2);
- **Redução de dimensionalidade** com TruncatedSVD (100 componentes);
- **Modelagem supervisionada** com XGBoost, otimizado via GridSearchCV;
- **Balanceamento de classes** utilizando scale_pos_weight ajustado dinamicamente.
""")

st.header("3. Resultados do Modelo Final")
st.markdown("""
**Melhores hiperparâmetros encontrados:**  
- learning_rate: 0.05  
- max_depth: 4  
- n_estimators: 100

**Métricas de desempenho:**

| Métrica              | Valor |
|----------------------|--------|
| AUC                  | 0.719  |
| F1-score (Classe 1)  | 0.55   |
| Recall (Classe 1)    | 0.56   |
| Precisão (Classe 1)  | 0.54   |
| Acurácia geral       | 0.68   |

> O modelo apresentou a melhor performance entre todas as versões testadas.
""")

st.header("4. Teste com Clusterização como Feature Adicional")
st.markdown("""
Foi testada a adição de um novo atributo categórico com base em **KMeans (n=4)** para verificar se a segmentação dos candidatos poderia melhorar a performance.

**Resultado:**
- AUC e F1-score permaneceram iguais ao modelo anterior
- Complexidade computacional aumentada sem ganho em performance

> Concluiu-se que a adição do cluster como feature não trouxe vantagem real.
""")

st.header("5. Pipeline Empacotada para Deploy")
st.markdown("""
A pipeline final foi empacotada com a biblioteca `joblib` para uso no ambiente de produção.

**Componentes da pipeline:**
- Classe de pré-processamento customizada (`TextoPreprocessador`)
- Vetorizador TF-IDF (1,2)
- Redutor SVD (100 componentes)
- Modelo XGBoost otimizado com hiperparâmetros ajustados

A pipeline pode ser carregada e utilizada diretamente para classificar novos textos.
""")