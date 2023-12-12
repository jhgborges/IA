from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# Função para carregar dados de um arquivo
def carregar_dados(arquivo):
    with open(arquivo, 'r', encoding='utf-8') as file:
        linhas = file.readlines()

    # Dividir as linhas em sentenças e rótulos
    sentencas, rotulos = zip(*[linha.strip().split(',') for linha in linhas])

    return sentencas, rotulos


# Carregar os dados de treinamento
arquivo_treinamento = 'knn.txt'
sentencas_treinamento, rotulos_treinamento = carregar_dados(arquivo_treinamento)

# Vetorizar as sentenças usando TF-IDF
vectorizer = TfidfVectorizer()
X_treinamento = vectorizer.fit_transform(sentencas_treinamento)

# Dividir os dados em conjunto de treinamento e conjunto de teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X_treinamento, rotulos_treinamento, test_size=0.2,
                                                        random_state=42)

# Criar o classificador kNN
knn_classificador = KNeighborsClassifier(n_neighbors=1, metric='cosine')
knn_classificador.fit(X_treino, y_treino)

# Sentença para classificar
#nova_sentenca = "eu gosto deste lugar"
nova_sentenca = "I always line foreign films"
nova_sentenca_vetorizada = vectorizer.transform([nova_sentenca])

# Classificar a nova sentença
predicao = knn_classificador.predict(nova_sentenca_vetorizada)

# Imprimir a classificação
print()
print(f"A sentença '{nova_sentenca}' é classificada como: {predicao[0]}")

# Avaliar a precisão do modelo no conjunto de teste
predicoes_teste = knn_classificador.predict(X_teste)
acuracia = accuracy_score(y_teste, predicoes_teste)
print(f'Acurácia no conjunto de teste: {acuracia:.2f}')

# Imprimir relatório de classificação
#print("\nRelatório de Classificação:")
#print(classification_report(y_teste, predicoes_teste))
