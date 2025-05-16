import joblib
import pandas as pd

def carregar_modelo(caminho_modelo='models/pipeline_modelo_nlp.pkl'):
    return joblib.load(caminho_modelo)

def prever_contratacao(modelo, texto):
    entrada = pd.Series([texto])
    pred = modelo.predict(entrada)
    prob = modelo.predict_proba(entrada)
    return pred[0], prob[0][1]
