import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# Carregar o arquivo CSV
file_path = 'notas_1100.csv'
data = pd.read_csv(file_path)
df = pd.DataFrame(data)

# Dividir os dados em features (X) e target (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]


# Dividir os dados em conjuntos de treinamento e teste (80% treinamento, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Inicializar o modelo de regressão logística
model = LogisticRegression()

# Treinar o modelo
model.fit(X_train, y_train)

X_test_named = pd.DataFrame(X_test, columns=X.columns)


# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test_named)

# Avaliar a acurácia do modelo
accuracy = accuracy_score(y_test, y_pred)
accuracy_porcentagem = "{:.2%}".format(accuracy)
print()
print(f'Acurácia do modelo: {accuracy_porcentagem}')

# Dado novo para classificação
new_data = [[1,3,5,7,4]]

# Fazer a previsão usando o modelo treinado
nova_classificacao = model.predict(new_data)

print(f'A classificação para os novos dados é: {nova_classificacao}')

if nova_classificacao == 0:
    print("Aluno REPROVADO!")
else:
    print("Aluno APROVADO!")

