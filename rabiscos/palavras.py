from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Substitua 'caminho_para_seu_arquivo.txt' pelo caminho para o seu arquivo de palavras
caminho_arquivo = 'br-sem-acentos.txt'

# Leitura do arquivo e armazenamento das palavras em uma lista
with open(caminho_arquivo, 'r', encoding='utf-8') as arquivo:
    linhas = arquivo.readlines()
    palavras = [linha.strip() for linha in linhas]

# Tokenização e pré-processamento
tokens = [word_tokenize(palavra.lower()) for palavra in palavras]

# Treinamento do modelo Word2Vec
modelo = Word2Vec(sentences=tokens, vector_size=100, window=5, min_count=1, workers=4)

# Salvar o modelo treinado
modelo.save("modelo_word2vec.bin")

# Agora, você pode usar o modelo para obter embeddings, encontrar palavras similares, etc.
