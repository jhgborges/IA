from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from docx2txt import process  # Certifique-se de instalar a biblioteca python-docx2txt: pip install python-docx2txt

# Substitua 'caminho_para_seu_modelo.bin' pelo caminho para o seu modelo Word2Vec pré-treinado
caminho_modelo = 'modelo_word2vec.bin'

# Substitua 'caminho_para_documento1.docx' e 'caminho_para_documento2.docx' pelos caminhos reais para seus documentos do Word
caminho_documento1 = 'a1.docx'
caminho_documento2 = 'a2.docx'

# Carregue o modelo Word2Vec pré-treinado
modelo = Word2Vec.load(caminho_modelo)

# Função para extrair texto de um documento do Word (.docx) usando python-docx2txt
def extrair_texto_do_docx(caminho_docx):
    return process(caminho_docx)

# Obtenha o texto de cada documento
texto_documento1 = extrair_texto_do_docx(caminho_documento1)
texto_documento2 = extrair_texto_do_docx(caminho_documento2)

# Função para obter embeddings médios de um texto
def obter_embeddings_medios(texto, modelo):
    tokens = word_tokenize(texto.lower())
    embeddings = [modelo.wv[palavra] for palavra in tokens if palavra in modelo.wv]
    if embeddings:
        return sum(embeddings) / len(embeddings)
    else:
        return None

# Obtenha os embeddings médios para cada documento
embedding_documento1 = obter_embeddings_medios(texto_documento1, modelo)
embedding_documento2 = obter_embeddings_medios(texto_documento2, modelo)

# Verifique se embeddings foram obtidos para ambos os documentos
if embedding_documento1 is not None and embedding_documento2 is not None:
    # Calcule a similaridade de cosseno entre os embeddings dos dois documentos
    similaridade = cosine_similarity([embedding_documento1], [embedding_documento2])[0][0]

    print(f'Similaridade entre os documentos: {similaridade}')
else:
    print('Não foi possível calcular a similaridade. Certifique-se de que o modelo e os documentos estão corretamente configurados.')
