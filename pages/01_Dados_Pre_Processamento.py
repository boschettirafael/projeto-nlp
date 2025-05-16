import streamlit as st

st.title("Dados e Pré-Processamento")

st.header("1. Carregamento e Normalização das Bases JSON")

st.markdown("""
Os dados utilizados foram fornecidos em arquivos JSON contendo informações sobre as vagas, os candidatos e suas respectivas situações nos processos seletivos.

Foi criada uma função genérica `carregar_e_processar_json` capaz de lidar com diferentes estruturas, normalizando listas e dicionários aninhados para a construção dos DataFrames:

- `df_vagas`: contendo os dados das vagas abertas
- `df_prospects`: contendo a situação dos candidatos em relação às vagas
- `df_applicants`: contendo os dados pessoais, acadêmicos e curriculares dos candidatos

Cada base foi posteriormente planificada por meio de funções específicas que extraem e desmembram os campos aninhados.
""")

st.code("""
def carregar_e_processar_json(nome_arquivo):
    caminho = Path('data/raw') / nome_arquivo
    with open(caminho, 'r', encoding='utf-8') as f:
        dados = json.load(f)
    ...
""", language="python")

st.header("2. Construção da Variável Target")

st.markdown("""
A variável `target` foi criada a partir da coluna `situacao_candidado` da base de prospects. Foram definidos dois grupos de rótulos:

- **Positivos**: candidatos contratados ou aprovados
- **Negativos**: candidatos reprovados ou recusados

Após essa filtragem, mantivemos apenas uma ocorrência por candidato (a mais recente).
""")

st.code("""
valores_positivos = ['Contratado pela Decision', 'Aprovado', 'Contratado como Hunting']
valores_negativos = ['Não Aprovado pelo Cliente', 'Não Aprovado pelo RH', ...]
df_filtrado['target'] = df_filtrado['situacao_candidado'].apply(lambda x: 1 if x in valores_positivos else 0)
""", language="python")

st.header("3. Unificação das Informações do Candidato")

st.markdown("""
Foi realizado um merge entre a base de candidatos e as informações de sua última participação em vaga, criando uma estrutura unificada contendo o `cv_pt` e o `target`.

Esse DataFrame foi chamado de `df_nlp`, sendo a base principal utilizada para os testes de vetorização e modelagem.
""")

st.header("4. Pré-processamento Textual (sem Stemming)")

st.markdown("""
Inicialmente foi criada uma versão de pré-processamento sem stemming, que incluiu:

- Normalização de caixa (minúsculas)
- Remoção de números, pontuação, símbolos especiais e palavras irrelevantes
- Tokenização e remoção de stopwords

O resultado foi armazenado na coluna `cv_pt_clean`.
""")

st.code("""
def preprocessar_sem_stemming(texto):
    texto = texto.lower()
    texto = re.sub(r'\\n|\\t', ' ', texto)
    texto = re.sub(r'\\d+', '', texto)
    ...
    return ' '.join(tokens)
""", language="python")

st.header("5. Vetorização TF-IDF")

st.markdown("""
Por fim, foi utilizada a vetorização TF-IDF com limite de 2000 features para transformar os textos em representações numéricas esparsas.

A variável `X` contém a matriz vetorizada e `y` contém os rótulos binários.
""")

st.code("""
vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(df_nlp_no_stem['cv_pt_clean'])
y = df_nlp_no_stem['target']
""", language="python")
