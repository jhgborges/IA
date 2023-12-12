import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Carregando o conjunto de dados Iris
iris = datasets.load_iris()
X = iris.data

# Padronizando os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicando o algoritmo KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Adicionando rótulos dos clusters ao conjunto de dados
iris_df = pd.DataFrame(data=X, columns=iris.feature_names)
iris_df['Cluster'] = y_kmeans

# Visualizando os clusters nos dois primeiros atributos
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, cmap='viridis', edgecolors='k', s=80)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('Clusters encontrados pelo KMeans')
plt.xlabel('Feature 1 (Padronizado)')
plt.ylabel('Feature 2 (Padronizado)')
plt.legend()
plt.show()
