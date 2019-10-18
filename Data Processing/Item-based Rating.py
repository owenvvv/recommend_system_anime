import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import random
from random import choice

anime = pd.read_excel("C:/Users/WU Shijun DD/Desktop/anime_common.xlsx")
Rating = pd.read_csv("C:/Users/WU Shijun DD/Desktop/rating.csv")

#print(Rating.head())
#print(Rating.rating.value_counts())

Rating.rating.value_counts().plot(kind='bar', title = 'Distribution Of Number of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

#print(Rating.info())
#print(Rating.isnull().sum())

ratings = Rating[:1000]
print(ratings.info())

from surprise import Reader, Dataset
reader = Reader()
data = Dataset.load_from_df(ratings[['user_id', 'anime_id', 'rating']], reader)

#Split into training and testing
print("Start Spliting Training and Testing")
from surprise.model_selection import train_test_split

trainset, testset = train_test_split(data, test_size=0.25)

print("Splitting Done")

from surprise import KNNBaseline,SVD, SVDpp,SlopeOne, NMF, NormalPredictor, KNNBasic, \
    KNNWithMeans,KNNWithZScore,CoClustering,BaselineOnly,accuracy
from surprise.model_selection import cross_validate

benchmark = []
#Iterate over all algorithms
for algorithm in [SVD(), SVDpp(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(), KNNBasic(), KNNWithMeans(),
                  KNNWithZScore(), BaselineOnly(), CoClustering()]:
    #Perform cross validation
    results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)

    #Get results & append algorithm name
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark.append(tmp)
    #print(tmp)

print(pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse'))

print("Build the Model")
sim_options = {"name": "cosine",
    'user_based': False}
algo = KNNBaseline(sim_options=sim_options)
print("Train the Model")
algo.fit(trainset)
print("Test the model")
predictions = algo.test(testset)
print(predictions)

from surprise import accuracy
print(accuracy.rmse(predictions))

def get_Iu(uid):
    """ return the number of items rated by given user
    args:
      uid: the id of the user
    returns:
      the number of items rated by the user
    """
    try:
        return len(trainset.ur[trainset.to_inner_uid(uid)])
    except ValueError:  # user was not part of the trainset
        return 0


def get_Ui(iid):
    """ return number of users that have rated given item
    args:
      iid: the raw id of the item
    returns:
      the number of users that have rated the item.
    """
    try:
        return len(trainset.ir[trainset.to_inner_iid(iid)])
    except ValueError:
        return 0

df = pd.DataFrame(predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])
df['Iu'] = df.uid.apply(get_Iu)
df['Ui'] = df.iid.apply(get_Ui)
df['err'] = abs(df.est - df.rui)
best_predictions = df.sort_values(by='err')[:10]
worst_predictions = df.sort_values(by='err')[-10:]

anime = anime.reset_index()
def index(Name):
    return anime[anime.Name == Name]["index"].values[0]

Testmovie = "Nana"
index_match = index(Testmovie)
print(index_match)
playlist_neighbors = algo.get_neighbors(index_match, k=10)

def name(index):
    return anime[anime.index == index]["Name"].values[0]
print('Animes Recommendation for similar movie '+ Testmovie + ':')
for x in range(0,10):
    print("No." + str(x + 1) + ": " + str(name(playlist_neighbors[x])))
