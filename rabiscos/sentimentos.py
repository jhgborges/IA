import nltk

def naive_bayes_sentiment_analysis(sentences, sentiments, smoothing=0.1):

    # Tokenize the sentences
    tokens = [nltk.word_tokenize(sentence) for sentence in sentences]

    # Create a vocabulary
    vocabulary = set()
    for tokens in tokens:
        for token in tokens:
            vocabulary.add(token)

    # Calculate the prior probabilities of the classes
    positive_probability = len(sentiments) * sentiments.count("positivo") / len(sentiments)
    negative_probability = len(sentiments) * sentiments.count("negativo") / len(sentiments)

    # Calculate the conditional probabilities of the words
    positive_word_probabilities = {}
    negative_word_probabilities = {}
    for token in vocabulary:
        positive_word_probabilities[token] = (
            len([sentence for sentence in tokens if token in sentence and sentiments[sentences.index(sentence)] == "positivo"])
            + smoothing
        ) / (len([sentence for sentence in tokens if token in sentence]) + smoothing)
        negative_word_probabilities[token] = (
            len([sentence for sentence in tokens if token in sentence and sentiments[sentences.index(sentence)] == "negativo"])
            + smoothing
        ) / (len([sentence for sentence in tokens if token in sentence]) + smoothing)

    # Classify the sentences
    for sentence, tokens in zip(sentences, tokens):
        positive_probability = 1
        negative_probability = 1
        for token in tokens:
            positive_probability *= positive_word_probabilities[token]
            negative_probability *= negative_word_probabilities[token]
        if positive_probability > negative_probability:
            sentiment = "positivo"
        else:
            sentiment = "negativo"
        print(sentence, sentiment)


# Colete um conjunto de dados de treinamento
sentences = [
    "Este filme é incrível!",
    "Eu não gostei desse restaurante",
    "O serviço foi muito bom",
    "O produto é de baixa qualidade"
]
sentiments = [
    "positivo",
    "negativo",
    "positivo",
    "negativo"
]

# Execute a análise de sentimento
naive_bayes_sentiment_analysis(sentences, sentiments, smoothing=0.1)