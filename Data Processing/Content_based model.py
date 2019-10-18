##### Imporing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

## Importing Textblob package
from textblob import TextBlob

# Importing CountVectorizer for sparse matrix/ngrams frequencies
from sklearn.feature_extraction.text import CountVectorizer

import nltk.compat
import itertools
import chardet
from sklearn.decomposition import TruncatedSVD         
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora,models,similarities

from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from pandas.core.frame import DataFrame
from collections import Counter
import csv
from textblob import Word

data = pd.DataFrame(pd.read_excel("synopsis.xlsx"))
Synopsis=data['Synopsis'].tolist()

#sentence tokenize
from nltk.tokenize import sent_tokenize
sent=[]
for i in Synopsis:
    sent.append(sent_tokenize(i))
#word_tokenize
from nltk.tokenize import word_tokenize
words=[]
for i in sent:
    for j in i:
        words.extend(word_tokenize(j))

#lower the cases
words_lower=[i.lower() for i in words]
#filter
english_stopwords = stopwords.words("english")
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '!', '@', '#', '%', '$', '*', '...','``',"'", "''"] 
english_other=["'s","n't",'l.']
words_clear=[]
for i in words_lower:
    if i not in english_stopwords: # stopwords
        if i not in english_punctuations: # english_punctuations
            if i not in english_other:
                words_clear.append(i)

from nltk.stem.porter import PorterStemmer
st = PorterStemmer()
words_stem=[st.stem(word) for word in words_clear]

#clear some meaningless common words
english_stop=["l.","one",'2022','howev','becom','must','day','year','11','c.c.','take','find','also','time','come',
             'first','name','make','take','known','follow','come','use','call','soon','begin','get','even','place',
             'turn','way','two','order','high','world','group','meet','live','one']
words_clear2=[]
for i in words_stem:
    if i not in english_stop: 
        words_clear2.append(i)

 #glance the most common words
from collections import Counter
words_counter=Counter(words_clear2)
words_counter.most_common(5)

# pick the adj and noun
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
from nltk.tag import pos_tag
pos_tag(words_clear2,tagset='universal')
AN=[]
for a,b in pos_tag(words_clear2,tagset='universal'):
    if b=="ADJ" or "Noun":
        AN.append(a)

#stem words
from nltk.stem.porter import PorterStemmer
st = PorterStemmer()
words_stem=[st.stem(word) for word in AN]

#remove rare words
freq = Counter(AN)
trim_words=[word for word in words if freq[word]>5]

#-----------------------------------------------------------------

text_2=data['Synopsis']
text_2 = pd.DataFrame(text_2)
text_2 = text_2.values.tolist()
#lower the cases
text_new=[]
for i in range(0,len(text_2)):
    a = str(text_2[i][0])
    text_new.append(a.lower())
#transform format
text_new_df=pd.DataFrame(text_new)
text_new_df=text_new_df[0]
text_3 = []
for i in text_new:
    a = []
    a.append(i)
    text_3.append(a)

#select the words
text_new_df = text_new_df.apply(lambda x: " ".join(x for x in x.split() if x in trim_words))

#construct dtm
training_samples = text_new_df.Synopsis[:len(text_new_df)]
dtm_vectorizer = CountVectorizer()
dtm = dtm_vectorizer.fit_transform(training_samples)
dtm.toarray()

#vectorize word 
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(training_samples)
terms = vectorizer.get_feature_names()

# set numbers of dimensions
n_pick_topics = 5            
lsa = TruncatedSVD(n_pick_topics)               
X2 = lsa.fit_transform(X)
df_new = pd.DataFrame(X2)
df_new_T=df_new.T

#calculate the similarity
cos_sim= []
for i in range(0,len(df_new)-1):
    cos_inner=[]
    for j in range(0,len(df_new)-1):
        cos1 = cosine(df_new_T[i],df_new_T[j])
        cos_inner.append(cos1)
    cos_sim.append(cos_inner)

data_sim=DataFrame(cos_sim)

#save the similarity matrix

data_sim.to_csv('/Users/zhoujingyu/Desktop/sim.csv')

#LDA
n_pick_topics = 5            # set numbers of dimensions
lsa = TruncatedSVD(n_pick_topics)               
X2 = lsa.fit_transform(X)

n_pick_docs=3
topic_docs_id = [X2[:,t].argsort()[:-(n_pick_docs+1):-1] for t in range(n_pick_topics)]
topic_docs_id

#key words of the topic
n_pick_keywords = 4
topic_keywords_id = [lsa.components_[t].argsort()[:-(n_pick_keywords+1):-1] for t in range(n_pick_topics)]

data_new = pd.DataFrame(pd.read_excel("/Users/zhoujingyu/Desktop/anime_common.xlsx"))
text_new=data['Synopsis']

for t in range(n_pick_topics):
    print("topic %d:" % t)
    print("    keywords: %s" % ", ".join(terms[topic_keywords_id[t][j]] for j in range(n_pick_keywords)))

for t in range(n_pick_topics):
    print("topic %d:" % t)
    print("    keywords: %s" % ", ".join(terms[topic_keywords_id[t][j]] for j in range(n_pick_keywords)))
    for i in range(n_pick_docs):
        print("    text %d" % i)
        print("\t"+text_new[topic_docs_id[t][i]])

