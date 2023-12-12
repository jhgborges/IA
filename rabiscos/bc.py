import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn import metrics
import matplotlib.pyplot as plt

# Carregando o conjunto de dados Breast Cancer Wisconsin (Diagnostic)
wdbc_df = pd.read_csv('wdbc.data', header=None)

# Selecionando as colunas relevantes (você pode ajustar isso com base em seu problema)
wdbc_df = wdbc_df.iloc[:, 1:]  # Ignorando a primeira coluna que contém os IDs dos pacientes
wdbc_df.columns = ['Diagnosis'] + [f'Feature_{i}' for i in range(1, 31)]

# Convertendo o diagnóstico para 0 (benigno) ou 1 (maligno)
wdbc_df['Diagnosis'] = wdbc_df['Diagnosis'].map({'M': 1, 'B': 0})

# Dividindo os dados em conjuntos de treinamento e teste (70% treinamento, 30% teste)
X = wdbc_df.drop('Diagnosis', axis=1)
y = wdbc_df['Diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criando o modelo Naive Bayes
model = GaussianNB()
#model = CategoricalNB()

# Treinando o modelo com os dados de treinamento
model.fit(X_train, y_train)

# Fazendo previsões com os dados de teste
y_pred = model.predict(X_test)

# Avaliando a precisão do modelo
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy:.2f}')

# Exibindo informações sobre o conjunto de dados
print(wdbc_df.info())

# Gráfico para visualizar as previsões do modelo
plt.figure(figsize=(10, 6))
plt.scatter(X_test['Feature_1'], X_test['Feature_2'], c=y_test, cmap='viridis', edgecolors='k', s=80, label='Real')
plt.scatter(X_test['Feature_1'], X_test['Feature_2'], c=y_pred, cmap='viridis', marker='x', s=80, linewidths=1, label='Previsto')
plt.title('Comparação entre Classes Reais e Previsões')
plt.xlabel('Feature_1')
plt.ylabel('Feature_2')
plt.legend()
plt.show()
