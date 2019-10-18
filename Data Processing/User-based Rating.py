import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import heapq

anime = pd.read_excel("C:/Users/WU Shijun DD/Desktop/anime_common.xlsx")
Rating = pd.read_csv("C:/Users/WU Shijun DD/Desktop/rating.csv")

#print(Rating.head())
#print(Rating.rating.value_counts())

Rating.rating.value_counts().plot(kind='bar')
#plt.show()

#print(Rating.info())
#print(Rating.isnull().sum())

ratings = Rating[:100000]
ratings['rating'] = ratings['rating'] / 2

from surprise import Reader, Dataset
reader = Reader()
data = Dataset.load_from_df(ratings[['user_id', 'anime_id', 'rating']], reader)

#Split into training and testing
print("Start Spliting Training and Testing")
from surprise.model_selection import train_test_split

trainset, testset = train_test_split(data, test_size=0.25)

print("Splitting Done")

from surprise import KNNBaseline, accuracy
print("Build the Model")
sim_options = {'name': 'cosine','user_based': True}
algo = KNNBaseline(sim_options=sim_options)
print("Train the Model")
algo.fit(trainset)
print("Test the model")
predictions = algo.test(testset)
#print(predictions)

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

# we can now query for specific predicions
uid = str(1)  # raw user id
iid = str(20)  # raw item id

# get a prediction for specific users and items.
pred = algo.predict(uid, iid, r_ui=4, verbose=True)
#print(pred)

#Get a list of all animes
animeid = Rating['anime_id'].unique()
#Get a list of animes that uid 50 has rated
animeid50 = Rating.loc[Rating['user_id'] == 50, 'anime_id']
#Remove the animes that uid 50 has rated
anime_to_predict = np.setdiff1d(animeid, animeid50)

testing = [[50, anime_id, 4.] for anime_id in anime_to_predict]
predictions = algo.test(testing)
predictions[0]

pred_ratings = np.array([pred.est for pred in predictions])
print(pred_ratings)
maxindex = np.where(pred_ratings == max(pred_ratings))
#print(maxindex)
max = heapq.nsmallest(10, maxindex[0])
#print(max)
def name(index):
    return anime[anime.index == index]["Name"].values[0]
print('Animes Recommendation for user 50:')
for x in range(0,10):
    iid = anime_to_predict[max[x]]
    print("No." + str(x+1) +" " + name(iid) + "|| Predicted Rate: " +str(round(pred_ratings[x],3)))

