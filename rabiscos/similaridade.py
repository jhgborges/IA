from gensim.models import Word2Vec

# Substitua 'caminho_para_seu_modelo.bin' pelo caminho para o seu modelo Word2Vec pré-treinado
caminho_modelo = 'modelo_word2vec.bin'

# Carregue o modelo Word2Vec pré-treinado
modelo = Word2Vec.load(caminho_modelo)

# Palavra de referência para calcular a similaridade
palavra1 = 'fisica'
palavra2 = 'quimica'

# Verifique se ambas as palavras estão no vocabulário do modelo
if palavra1 in modelo.wv and palavra2 in modelo.wv:
    # Calcule a similaridade entre as duas palavras
    similaridade = modelo.wv.similarity(palavra1, palavra2)

    print(f'Similaridade entre "{palavra1}" e "{palavra2}": {similaridade}')
else:
    print(f'Uma ou ambas as palavras não estão no vocabulário do modelo.')

