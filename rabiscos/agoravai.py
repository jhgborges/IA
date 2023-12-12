import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Leitura do arquivo CSV (substitua 'seu_arquivo.csv' pelo nome do seu arquivo)
nome_arquivo = 'datase1.csv'
dados = pd.read_csv(nome_arquivo)

# Exibição das primeiras linhas do arquivo para verificar os dados
print("Primeiras linhas do arquivo:")
print(dados.head())

# Seleção das colunas necessárias
colunas_selecionadas = ['coluna1', 'coluna2']  # Substitua pelos nomes reais das colunas
X = dados[colunas_selecionadas]

# Escalonamento de features (opcional, mas recomendado para o KMeans)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Número de clusters desejado
num_clusters = 3  # Substitua pelo número desejado de clusters

# Aplicação do algoritmo KMeans
modelo_kmeans = KMeans(n_clusters=num_clusters, random_state=0)
modelo_kmeans.fit(X_scaled)

# Adicionando rótulos ao DataFrame original
dados['rotulo'] = modelo_kmeans.labels_

# Visualização dos clusters
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=modelo_kmeans.labels_, cmap='rainbow')
plt.scatter(modelo_kmeans.cluster_centers_[:, 0], modelo_kmeans.cluster_centers_[:, 1], s=100, c='black', marker='X')
plt.title('Clusters gerados pelo KMeans')
plt.xlabel('Coluna 1')
plt.ylabel('Coluna 2')

# Definindo os limites dos eixos com base nos valores máximos e mínimos das colunas
plt.xlim(X.iloc[:, 0].min(), X.iloc[:, 0].max())
plt.ylim(X.iloc[:, 1].min(), X.iloc[:, 1].max())

plt.show()
