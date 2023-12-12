import pandas as pd
import warnings
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Inicia a medição do tempo de execução do código
inicio = (time.time())

# Carregar o arquivo CSV
file_path = 'notas_1100.csv'
data = pd.read_csv(file_path)

# Dividir os dados em features (X) e target (y)
X = data.iloc[:, :-1]  # Todas as colunas, exceto a última
y = data.iloc[:, -1]   # Última coluna

# Dividir os dados em conjuntos de treinamento e teste (80% treinamento, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicializar o modelo de regressão logística
model = LogisticRegression()

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar a acurácia do modelo
accuracy = accuracy_score(y_test, y_pred)
accuracy_porcentagem = "{:.2%}".format(accuracy)
print()
print(f'Acurácia do modelo: {accuracy_porcentagem}')


# Dado novo para classificação
new_data = [[1,3,5,7,4]]

# Fazer a previsão usando o modelo treinado
nova_classificacao = model.predict(new_data)

# Calcular a matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)

fim = time.time()

print()
print(f"Tempo gasto: {fim - inicio:.2} segundos!")

print(f'A classificação para os novos dados é: {nova_classificacao}')

if nova_classificacao == 0:
    print("Aluno REPROVADO!")
else:
    print("Aluno APROVADO!")

# Visualizar a matriz de confusão usando um mapa de calor
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Reprovado', 'Aprovado'], yticklabels=['Reprovado', 'Aprovado'])
plt.xlabel('Previsto')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão para Regreesão Logística')

# Exibir o gráfico no ambiente interativo
plt.show()
