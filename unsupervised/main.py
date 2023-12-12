import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Leitura do arquivo CSV (substitua 'seu_arquivo.csv' pelo caminho do seu arquivo CSV)
data = pd.read_csv('c1.csv')

# Suponha que o arquivo CSV contenha colunas 'feature1' e 'feature2'.
# Você pode ajustar esses nomes de coluna de acordo com o seu arquivo.
X = data[['distancia']]
Y = data[['classe']]

# Escolha o número de clusters (grupos) desejado
n_clusters = 3

# Criar um modelo K-Means
kmeans = KMeans(n_clusters=n_clusters)

# Ajustar o modelo aos dados
kmeans.fit(X)

# Obter os rótulos de cluster para cada ponto de dados
labels = kmeans.labels_

# Adicionar os rótulos de cluster de volta ao DataFrame
data['cluster'] = labels

# Visualizar os resultados do agrupamento
plt.scatter(X['distancia'], Y['classe'], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], [0] * n_clusters, s=300, c='red', label='Centroids')
plt.xlabel('Distancia')
plt.ylabel('classe')
plt.legend()
plt.show()

# Exibir os centróides dos clusters
print("Centróides dos Clusters:")
print(kmeans.cluster_centers_)

