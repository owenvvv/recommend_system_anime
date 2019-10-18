import pandas as pd
import numpy as np
import scipy as sp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

anime = pd.read_excel("C:/Users/WU Shijun DD/Desktop/anime_common.xlsx")

anime = anime[:1000]
features = ['Genre','Studio']

def combination(row):
    return row['Genre']+" "+row['Studio']

for feature in features:
    anime[feature] = anime[feature].fillna('') #filling all NaNs with blank string

anime['combination'] = anime.apply(combination,axis=1)

count = CountVectorizer()
count_matrix = count.fit_transform(anime['combination'])

# Compute the Cosine Similarity matrix based on the count_matrix
cosine_sim2 = cosine_similarity(count_matrix)
print(cosine_sim2)

# Reset index of our main DataFrame and construct reverse mapping as before
anime = anime.reset_index()

def name(index):
    return anime[anime.index == index]["Name"].values[0]
def studio(index):
    return anime[anime.index == index]["Studio"].values[0]
def genre(index):
    return anime[anime.index == index]["Genre"].values[0]
def index(Name):
    return anime[anime.Name == Name]["index"].values[0]

user_input = "Nana"
index_match = index(user_input)
similar_animes = list(enumerate(cosine_sim2[index_match]))
final_list = sorted(similar_animes,key=lambda x:x[1],reverse=True)[1:]
i=0
print("Top 10 animes similar to '"+user_input+"' are:\n")
for x in final_list:
    id = x[0]
    print("No." + str(i+1) + ": " + name(id))
    i=i+1
    if i>9:
        break