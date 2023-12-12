import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Carregando o arquivo CSV
file_path = 'notas_1100.csv'  # Substitua com o caminho do seu arquivo CSV
df = pd.read_csv(file_path)

# Dividindo os dados em features (X) e target (y)
X = df.iloc[:, :-1]  # Todas as colunas, exceto a última
y = df.iloc[:, -1]   # Última coluna

# Dividindo os dados em conjuntos de treinamento (70%) e teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criando o modelo de árvore de decisão
model = DecisionTreeClassifier()

# Treinando o modelo
model.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliando a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print()
print(f'Acurácia do modelo: {accuracy:.2f}\n')

# Exibindo relatório de classificação
print('Relatório de Classificação:\n', classification_report(y_test, y_pred))

# Visualizando a matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
