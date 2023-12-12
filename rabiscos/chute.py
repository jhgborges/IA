from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import word_tokenize #Used to extract words from documents
from nltk.stem import WordNetLemmatizer #Used to lemmatize words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
import nltk
import wordcloud

from sklearn.cluster import KMeans

import sys
from time import time

import pandas as pd
import numpy as np
#nltk.download('wordnet')

categories = [
    'comp.graphics',
    'sci.med'
]

# df = DataFrames
df = fetch_20newsgroups(subset='all', categories=categories, shuffle=False, remove=('headers', 'footers', 'quotes'))

labels = df.target
true_k = len(np.unique(labels))
#print(true_k)

lemmatizer = WordNetLemmatizer()
for i in range(len(df.data)):
    word_list = word_tokenize(df.data[i])
    lemmatized_doc = ""
    for word in word_list:
        lemmatized_doc = lemmatized_doc + " " + lemmatizer.lemmatize(word)
    df.data[i] = lemmatized_doc

print(df.data[0])

# Remoção de stopwords em inglês
vectorizer = TfidfVectorizer(strip_accents='unicode', stop_words='english', min_df=2)
X = vectorizer.fit_transform(df.data)

# Agrupa os dados em clusters
km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100)
t0 = time()
km.fit(X)
print("Finalizado em %0.3fs" % (time() - t0))

centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()
for i in range(true_k):
    print("Cluster %d e suas 5 palavras mais frequentes: " % i, end='')
    for ind in centroids[i, :5]:
        print(' %s' % terms[ind], end='')
    print()

from wordcloud import WordCloud
import matplotlib.pyplot as plt

def frequencies_dict(cluster_index):
    if cluster_index > true_k - 1:
        return
    term_frequencies = km.cluster_centers_[cluster_index]
    sorted_terms = centroids[cluster_index]
    frequencies = {terms[i]: term_frequencies[i] for i in sorted_terms}
    return frequencies

def makeImage(frequencies):

    # Define a nuvem de palavras
    wc = WordCloud(background_color="white", max_words=50)

    # gera a nuvem de palavras
    wc.generate_from_frequencies(frequencies)

    # exibe a nuvem de palavras
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()

# Exibe os clusters sequencialmente
for i in range(true_k):
    freq = frequencies_dict(i)
    makeImage(freq)
