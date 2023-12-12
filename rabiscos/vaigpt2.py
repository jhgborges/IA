import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import nltk
from nltk.corpus import stopwords

# Baixar a lista de stopwords em português
nltk.download('stopwords')
stopwords_portugues = set(stopwords.words('portuguese'))

# Ler o arquivo CSV
# Substitua 'arquivo.csv' pelo caminho real do seu arquivo CSV
df = pd.read_csv("ex2.csv", encoding='utf-8')

# Certifique-se de que as colunas estão presentes no DataFrame
if 'sentenca' in df.columns and 'rotulo' in df.columns:
    # Dividir os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(df['sentenca'], df['rotulo'], test_size=0.2, random_state=42)

    # Criar um pipeline com vetorização de texto e o classificador Naive Bayes
    model = make_pipeline(CountVectorizer(stop_words=list(stopwords_portugues)), MultinomialNB())

    # Treinar o modelo
    model.fit(X_train, y_train)

    # Avaliar o modelo nos dados de teste
    y_pred = model.predict(X_test)

    # Imprimir métricas
    print("Métricas de classificação:")
    print(metrics.classification_report(y_test, y_pred))
else:
    print("Certifique-se de que as colunas 'sentenca' e 'rotulo' estão presentes no seu DataFrame.")
