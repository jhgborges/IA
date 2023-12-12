import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Baixar a lista de stopwords para o idioma desejado (neste caso, português)
#nltk.download('stopwords')
#nltk.download('punkt')

def remover_stopwords(texto):
    # Tokenizar o texto em palavras
    palavras = word_tokenize(texto)

    # Obter a lista de stopwords em português
    stopwords_portugues = set(stopwords.words('portuguese'))

    # Remover as stopwords do texto
    palavras_sem_stopwords = [palavra for palavra in palavras if palavra.lower() not in stopwords_portugues]

    # Reunir as palavras novamente em um texto
    texto_sem_stopwords = ' '.join(palavras_sem_stopwords)

    return texto_sem_stopwords

# Exemplo de uso
texto_com_stopwords = "Este é um exemplo de texto com algumas stopwords em português."
texto_sem_stopwords = remover_stopwords(texto_com_stopwords)

print("Texto original:")
print(texto_com_stopwords)

print("\nTexto sem stopwords:")
print(texto_sem_stopwords)
