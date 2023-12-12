# Importe a biblioteca necessária
import pandas as pd

# Abra o arquivo
data = pd.read_csv("ex2.csv")

X = data.iloc[:, :-1]

# Crie uma função para converter as sentenças em números
def to_numbers(sentence):
    words = sentence.split()
    vectors = []
    for word in words:
        vector = model.predict(word)
        vectors.append(vector)
    return sum(vectors)

#X = X.apply(to_numbers)

# Itere sobre as sentenças
for sentence in data["sentence"]:
    # Aplique a função
    converted_sentence = to_numbers(sentence)

# Faça algo com a sentença convertida
print(converted_sentence)
