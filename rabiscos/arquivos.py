from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

# Substitua 'caminho_para_seu_modelo.bin' pelo caminho para o seu modelo Word2Vec pré-treinado
caminho_modelo = "modelo_word2vec.bin"

# Substitua 'caminho_para_arquivo1.txt' e 'caminho_para_arquivo2.txt' pelos caminhos reais para os seus arquivos de texto
caminho_arquivo1 = "a1.c"
caminho_arquivo2 = "a2.c"

# Carregue o modelo Word2Vec pré-treinado
modelo = Word2Vec.load(caminho_modelo)

# Função para obter embeddings médios de um texto
def obter_embeddings_medios(texto, modelo):
    tokens = word_tokenize(texto.lower())
    embeddings = [modelo.wv[palavra] for palavra in tokens if palavra in modelo.wv]
    if embeddings:
        return sum(embeddings) / len(embeddings)
    else:
        return None

# Leia o conteúdo dos arquivos
with open(caminho_arquivo1, 'r', encoding='utf-8') as arquivo:
    texto_arquivo1 = arquivo.read()

with open(caminho_arquivo2, 'r', encoding='utf-8') as arquivo:
    texto_arquivo2 = arquivo.read()

# Obtenha os embeddings médios para cada arquivo
embedding_arquivo1 = obter_embeddings_medios(texto_arquivo1, modelo)
embedding_arquivo2 = obter_embeddings_medios(texto_arquivo2, modelo)

# Verifique se embeddings foram obtidos para ambos os arquivos
if embedding_arquivo1 is not None and embedding_arquivo2 is not None:
    # Calcule a similaridade de cosseno entre os embeddings dos dois arquivos
    similaridade = cosine_similarity([embedding_arquivo1], [embedding_arquivo2])[0][0]

    print(f'Similaridade entre os arquivos: {similaridade}')
else:
    print('Não foi possível calcular a similaridade. Certifique-se de que o modelo e os arquivos estão corretamente configurados.')
