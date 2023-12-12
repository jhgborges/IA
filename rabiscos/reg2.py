import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Substitua 'caminho/do/seu/arquivo.csv' pelo caminho real do seu arquivo CSV
file_path = 'notas_1100.csv'

# Carregar o arquivo CSV
data = pd.read_csv(file_path)

# Dividir os dados em features (X) e target (y)
X = data.iloc[:, :-1]  # Todas as colunas, exceto a última
y = data.iloc[:, -1]   # Última coluna

# Dividir os dados em conjuntos de treinamento e teste (70% treinamento, 30% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicializar o modelo de regressão logística
model = LogisticRegression()

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar a acurácia do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy}')

# Visualização dos resultados em modo texto
print("\nPrevisões do Modelo:")
df_resultados = pd.DataFrame({'Real': y_test, 'Previsto': y_pred})
print(df_resultados)

# Visualização em um gráfico de dispersão
plt.scatter(range(len(y_test)), y_test, color='blue', label='Real', marker='o')
plt.scatter(range(len(y_test)), y_pred, color='red', label='Previsto', marker='x')
plt.title('Comparação entre Valores Reais e Previstos')
plt.xlabel('Amostras')
plt.ylabel('Classes')
plt.legend()
plt.show()
