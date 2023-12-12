import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Leitura do arquivo CSV
df = pd.read_csv('notas_1100.csv')

# Selecionando as colunas relevantes (por exemplo, as colunas 1 a 5)
data = df.iloc[:, 1:5]

# Número de clusters desejado
num_clusters = 2

# Aplicando o algoritmo KMeans
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(data)

# Adicionando uma nova coluna ao DataFrame indicando o cluster de cada ponto
df['cluster'] = kmeans.labels_

# Visualizando os clusters
plt.scatter(df['n1'], df['n2'], c=df['cluster'], cmap='rainbow')
plt.title('Clusters KMeans')
plt.xlabel('Coluna1')
plt.ylabel('Coluna2')
plt.show()
