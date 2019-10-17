import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer

from movie_recommend.models import AnimeData
from django.core.cache import cache

import os

BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
N_MAX_WORDS = 30000

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer

# tknzr = WordPunctTokenizer()
# nltk.download('stopwords')
# stoplist = stopwords.words('english')
# from nltk.stem.porter import PorterStemmer

# stemmer = PorterStemmer()

'''
def PreprocessTfidf(texts, stoplists=[], stem=False):
    newtexts = []
    for i in range(len(texts)):
        text = texts[i]
        if stem:
            tmp = [w for w in tknzr.tokenize(text) if w not in stoplist]
        else:
            tmp = [stemmer.stem(w) for w in tknzr.tokenize(text) if w not in stoplist]
        newtexts.append(' '.join(tmp))
    return newtexts
'''


def load_data():
    cos_gerne = pd.read_csv("sim_gerne.csv", index_col=0)
    cos_dtm = pd.read_csv("sim_dtm.csv", index_col=0)
    uu_matrix = pd.read_csv("uu_matrix.csv", index_col=0)
    cluster = pd.read_csv("cluster.csv", index_col=0)
    cache.set('cos_gerne', cos_gerne)
    cache.set('cos_dtm', cos_dtm)
    cache.set('uu_matrix', uu_matrix)
    cache.set('cluster', cluster)


def load_anime():
    data = pd.DataFrame(pd.read_excel("anime_common.xlsx"))
    titles = data['Name'].to_list()
    genres = data['Genre'].to_list()
    types = data['Type'].to_list()
    statuses = data['Status'].to_list()
    sdudios = data['Studio'].to_list()
    durations = data['Duration'].to_list()
    categories = data['Category'].to_list()
    images = data['Image'].to_list()
    lenanime = len(titles[:])
    for m in range(lenanime):
        animedata = AnimeData()
        animedata.title = titles[m]
        animedata.type = types[m]
        animedata.duration = durations[m]
        animedata.status = statuses[m]
        animedata.imagepath = images[m]
        animedata.category = categories[m]
        animedata.studio = sdudios[m]
        animedata.genre = genres[m]
        animedata.animeID = data.ID[m]
        animedata.rating = data.Rating[m]
        animedata.description = json.dumps(data.Synopsis[m])
        animedata.save()


if __name__ == "__main__":
    load_data()
