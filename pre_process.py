import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download("punkt")
nltk.download("stopwords")


# Função para pré-processamento de texto
def preprocess_text(text):
    # Remover pontuações e caracteres especiais
    text = re.sub(r"[^\w\s]", "", text)
    # Converter para minúsculas
    text = text.lower()
    # Tokenização
    tokens = word_tokenize(text)
    # Remover stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    # Stemming ou Lemmatização (opcional)
    # stemmer = PorterStemmer()
    # tokens = [stemmer.stem(word) for word in tokens]
    # Juntar tokens de volta em uma string
    processed_text = " ".join(tokens)
    return processed_text
