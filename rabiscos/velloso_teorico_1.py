import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
#from nltk.probability import FreqDist
from nltk.classify import NaiveBayesClassifier

#nltk.download('stopwords')

# Conjunto de treinamento
training_data = [
    ("eu não gosto deste restaurante", "Negativa"),
    ("estou cansado dessas coisas", "Negativa"),
    ("eu me sinto bem com essas cervejas", "Positiva"),
    ("eu amo esse sanduíche", "Positiva"),
    ("este é um lugar incrível!", "Positiva"),
]

# Função para pré-processamento
def preprocess(sentence):
    stop_words = set(stopwords.words('portuguese'))
    #print(len(stop_words))
    words = word_tokenize(sentence.lower())
    print(words)
    return {word: True for word in words if word.isalnum() and word not in stop_words}

def remove_stopwords(sentence):
    stopwords = nltk.corpus.stopwords.words('portuguese')
    words = nltk.word_tokenize(sentence)
    return [word for word in words if word not in stopwords]

# Criar lista de tuplas (palavras, classe)
training_set = [(preprocess(sentence), label) for sentence, label in training_data]

# Criar o classificador Naive Bayes
classifier = NaiveBayesClassifier.train(training_set)
print(classifier.show_most_informative_features(100))
#print(nltk.classify.accuracy(classifier, training_set))

# Sentença para classificar
nova_sentenca = "eu gosto deste lugar"
filtrada = remove_stopwords(nova_sentenca)
nova_sentenca_preprocessada = preprocess(nova_sentenca)

print(nova_sentenca_preprocessada)

# Classificar a nova sentença
classe_predita = classifier.classify(nova_sentenca_preprocessada)

# Imprimir a classe predita
print(f"A sentença é classificada como: {classe_predita}")