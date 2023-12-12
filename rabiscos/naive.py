import pandas as pd
import time
import warnings
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Inicia a medição do tempo de execução do código
inicio = (time.time())

# 1. Ler o arquivo CSV
nome_arquivo = 'notas_1100.csv'
dados = pd.read_csv(nome_arquivo)

# 2. Dividir os dados em características (X) e rótulos (y)
X = dados.drop('status', axis=1)  # Substitua 'Rótulo' pelo nome da coluna que contém os rótulos
y = dados['status']

# 3. Dividir os dados em conjunto de treinamento e teste (70% treinamento, 30% teste)
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Criar e treinar o modelo Naive Bayes
modelo_nb = GaussianNB()
modelo_nb.fit(X_treino, y_treino)

# 5. Fazer previsões no conjunto de teste
previsoes = modelo_nb.predict(X_teste)

# 6. Avaliar a precisão do modelo
acuracia = metrics.accuracy_score(y_teste, previsoes)
acuracia_formatada = f"{acuracia: .2%}"

# Dado novo para classificação
new_data = [[1,3,5,7,4]]

# Fazer a previsão usando o modelo treinado
nova_classificacao = modelo_nb.predict(new_data)
print()
print(f'A classificação para os novos dados é: {nova_classificacao}')

fim = time.time()

print()
print(f"Tempo gasto: {fim - inicio:.2} segundos!")
print(f'Acurácia do modelo: {acuracia_formatada}')

relatorio_classificacao = metrics.classification_report(y_teste, previsoes)
print('Relatório de Classificação:')
print(relatorio_classificacao)



# 7. Apresentar resultados da classificação em modo gráfico
matriz_confusao = metrics.confusion_matrix(y_teste, previsoes)
sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues', xticklabels=modelo_nb.classes_, yticklabels=modelo_nb.classes_)
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

