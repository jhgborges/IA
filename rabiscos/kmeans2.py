import pandas as pd
import time
import warnings
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Carregar o arquivo CSV
# Suponha que o arquivo CSV tenha o nome 'dados.csv'
file_path = 'notas_1100.csv'
df = pd.read_csv(file_path)

# Suprimir a última coluna (target)
X = df.iloc[:, :-1]

# Separar os dados em conjuntos de treinamento (80%) e teste (20%)
X_train, X_test = train_test_split(X, test_size=0.2)

# Criar e treinar o modelo KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train)

# Adicionar as previsões ao conjunto de treinamento
X_train['Cluster'] = kmeans.predict(X_train)

# Exibir resultados em modo texto
print("Resultados de Classificação:")
print(X_train)

# Exibir resultados em modo gráfico
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.scatterplot(x='n2', y='media', hue='Cluster', data=X_train, palette='viridis', legend='full')
plt.title('Resultado da Classificação com KMeans')
plt.show()
