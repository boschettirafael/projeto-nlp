# src/preprocessing.py
from sklearn.base import BaseEstimator, TransformerMixin
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

nltk.download('stopwords')

stopwords_pt = set(stopwords.words('portuguese'))
stemmer = RSLPStemmer()

def preprocessar_com_stemming(texto):
    texto = texto.lower()
    texto = re.sub(r"[^a-zA-Zà-úÀ-Ú\s]", " ", texto)
    palavras = texto.split()
    palavras = [stemmer.stem(p) for p in palavras if p not in stopwords_pt]
    return " ".join(palavras)

class TextoPreprocessador(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.apply(preprocessar_com_stemming)
