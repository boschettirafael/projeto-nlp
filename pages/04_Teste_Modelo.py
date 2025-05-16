import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from src.model_utils import carregar_modelo, prever_contratacao

# Configuração visual e título da página
st.set_page_config(page_title="Simulação de Candidatos", layout="centered", page_icon="🧪")
st.title("🧪 Simulação com Exemplos Reais")

# Explicação inicial
st.markdown("""
Este projeto utiliza Machine Learning para prever automaticamente se um candidato tem alto potencial de contratação com base em seu histórico textual, extraído de currículos e dados associados às vagas.

A previsão é feita por meio de um modelo treinado sobre decisões anteriores da empresa, e retorna uma **classificação (aprovado/reprovado)** com um grau de **confiança (probabilidade)**.
""")

# Carrega modelo e base de exemplos
modelo = carregar_modelo()
df_exemplos = pd.read_csv('data/exemplos_simulacao.csv')

st.header("Sobre os candidatos listados")
st.markdown("""
Os currículos abaixo são **exemplos reais da base**, selecionados aleatoriamente entre candidatos com histórico conhecido (aprovado ou não).

Você pode explorar diferentes perfis e observar como o modelo reage em termos de classificação e confiança.
""")

# Escolha do candidato
codigo_escolhido = st.selectbox("Escolha um candidato:", df_exemplos['codigo'])
texto = df_exemplos.loc[df_exemplos['codigo'] == codigo_escolhido, 'texto_completo'].values[0]

# Exibe o texto
st.text_area("Texto do currículo selecionado:", texto, height=250)

st.header("Resultado da classificação")
st.markdown("""
Ao clicar no botão abaixo, o modelo irá analisar o conteúdo textual e emitir uma **previsão binária** com base no aprendizado de padrões anteriores.

- **Confiança** refere-se à **probabilidade prevista da classe 'aprovado'** (valor entre 0 e 1).
- Esta **não é uma decisão definitiva**, e sim um apoio estratégico ao processo de triagem.
""")

# Botão de classificação
if st.button("Classificar candidato"):
    resultado, probabilidade = prever_contratacao(modelo, texto)
    if resultado == 1:
        st.success(f"✅ Candidato aprovado! Confiança: {probabilidade:.2f}")
    else:
        st.error(f"❌ Candidato reprovado. Confiança: {probabilidade:.2f}")
