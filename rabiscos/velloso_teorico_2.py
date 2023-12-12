from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import numpy as np

# Conjunto de treinamento
training_data = [
    ("Just plain boring", "Negativo"),
    ("Entirely predictable and lacks energy", "Negativo"),
    ("No surprises and very few laughs", "Negativo"),
    ("Very powerful", "Positivo"),
    ("The most fun film of the summer", "Positivo"),
]

# Separar os textos e as classes
textos, classes = zip(*training_data)

# Vetorizar os textos usando TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(textos)

# Sentença a ser classificada
nova_sentenca = "I always like foreign films"
nova_sentenca_vec = vectorizer.transform([nova_sentenca])

# Classificar com kNN (k = 1) usando distância cosseno
knn_cosseno = KNeighborsClassifier(n_neighbors=1, metric='cosine')
knn_cosseno.fit(X, classes)
predicao_cosseno = knn_cosseno.predict(nova_sentenca_vec)[0]

# Calcular a distância cosseno entre a nova sentença e os exemplos de treinamento
distancias_cosseno = cosine_distances(nova_sentenca_vec, X)

# Classificar com kNN (k = 1) usando distância euclidiana
knn_euclidiana = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
knn_euclidiana.fit(X, classes)
predicao_euclidiana = knn_euclidiana.predict(nova_sentenca_vec)[0]

# Calcular a distância euclidiana entre a nova sentença e os exemplos de treinamento
distancias_euclidiana = euclidean_distances(nova_sentenca_vec, X)

# Imprimir resultados
print(f"Classificação usando distância cosseno: {predicao_cosseno}")
print(f"Distâncias cosseno para exemplos de treinamento: {distancias_cosseno}")
print(f"Classificação usando distância euclidiana: {predicao_euclidiana}")
print(f"Distâncias euclidianas para exemplos de treinamento: {distancias_euclidiana}")
