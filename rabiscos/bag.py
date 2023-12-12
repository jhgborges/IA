import re
import nltk
import string

# Importa os módulos necessários para uso da Natural Language Tool Kit
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize

def to_lowercase_list(words): # Converte texto para caixa baixa
    return [word.lower() for word in words]

def remover_pontuacao(texto): # remove puntuaçào do texto
    translator = str.maketrans('', '', string.punctuation)
    texto_sem_pontuacao = texto.translate(translator)
    return texto_sem_pontuacao

# Abre o arquivo escolhido como 1
arquivo1 = open("rec.motorcycles.txt", "r", encoding="utf8")

# Abre o arquivo escolhido como 2
arquivo2 = open("soc.religion.christian.txt", "r", encoding="utf8")

# Converte o conteúdo do arquivo para uma lista de palavras
palavras = arquivo1.read().split()
palavras2 = arquivo2.read().split()

# Aplica a funçào que converte o texto para caixa baixa
minusculas = to_lowercase_list(palavras)
minusculas2 = to_lowercase_list(palavras2)

# Remove a pontuação do texto
for minuscula in minusculas:
    minuscula_sem_pontos = remover_pontuacao(minuscula)

for minuscula2 in minusculas2:
    minuscula2_sem_pontos = remover_pontuacao(minuscula2)

# Define as stopwords em inglês
stopwords = set(stopwords.words('english'))

# Remove as stopwords em inglês
minusculas_limpas = [minuscula for minuscula in minusculas if minuscula.lower() not in stopwords]
minusculas2_limpas = [minuscula2 for minuscula2 in minusculas2 if minuscula2.lower() not in stopwords]

# Calcula a distribuição de frequência
freq_dist = FreqDist(minusculas_limpas)

# Mostra as 20 palavras mais frequentes no arquivo
bag_words = freq_dist.most_common(20)

# Calcula a distribuiçào de frequências para o arquivo 2
freq_dist2 = FreqDist(minusculas2_limpas)
bag_words2 = freq_dist2.most_common(20)

# Conta o número de vezes que cada palavra aparece na lista
contagem = {}
for palavra in minusculas_limpas:
    if palavra not in contagem:
        contagem[palavra] = 1
    else:
        contagem[palavra] += 1

# Conta o número de vezes que cada palavra aparece na lista
contagem2 = {}
for palavra in minusculas2_limpas:
    if palavra not in contagem2:
        contagem2[palavra] = 1
    else:
        contagem2[palavra] += 1

# Armazena os resultados em um dicionário
bag_of_words = contagem
bag_of_words2 = contagem2

#print(bag_of_words)
print(bag_words)
print(bag_words2)
print("--------------------#####--------------------")
print(bag_of_words)
print(bag_of_words2)

# Fecha os arquivos
arquivo1.close()
arquivo2.close()
