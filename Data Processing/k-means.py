import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


anime = pd.read_excel('/Users/xu/Desktop/anime_common.xlsx')
anime.isnull().sum()
anime["Rating"] = anime["Rating"].astype(float)
anime["members"] = anime["members"].astype(float)
anime["Name"] = anime["Name"].map(lambda name:re.sub('[^A-Za-z0-9]+', " ", name))
anime_features = pd.concat([anime["Genre"].str.get_dummies(sep=", "),
                            pd.get_dummies(anime[["Type"]]),
                            pd.get_dummies(anime[["Studio"]]),
                            pd.get_dummies(anime[["Studio"]]),
                            pd.get_dummies(anime[["Status"]]),
                            pd.get_dummies(anime[["Category"]]),
                            anime[["Rating"]]], axis=1)
anime_features.head()


#standardize
min_max_scaler = MinMaxScaler()
anime_features = min_max_scaler.fit_transform(anime_features)
np.round(anime_features,2)


#dimensionality reduction using PCA
pca = PCA(n_components=3)
pca.fit(anime_features)
pca_samples = pca.transform(anime_features)
ps = pd.DataFrame(pca_samples)
print(ps.head())

# plot 3-dimension clustering result
tocluster = pd.DataFrame(ps[[0,1,2]])
plt.rcParams['figure.figsize'] = (16, 9)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(tocluster[0], tocluster[2], tocluster[1])
plt.title('Data points in 3D PCA axis', fontsize=20)
plt.show()


# using SSE method to identify k-value in k-means
SSE = []  # create a set
for k in range(1,9):
    estimator = KMeans(n_clusters=k)
    estimator.fit(anime_features)
    SSE.append(estimator.inertia_)
X = range(1,9)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(X,SSE,'o-')
plt.show()

# using score method to identify k-value in k-means
scores = []
inertia_list = np.empty(8)
for i in range(2,8):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(tocluster)
    inertia_list[i] = kmeans.inertia_
    scores.append(silhouette_score(tocluster, kmeans.labels_))
plt.plot(range(2,8), scores);
plt.title('Results KMeans')
plt.xlabel('n_clusters');
plt.axvline(x=4, color='red', linestyle='--')
plt.ylabel('Silhouette Score');
plt.show()

# combining 2 method, 4 is the most efficient clustering
clusterer = KMeans(n_clusters = 4, random_state = 30).fit(tocluster)
centers = clusterer.cluster_centers_
print(centers)
c_preds = clusterer.predict(tocluster)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(tocluster[0], tocluster[2], tocluster[1], c = c_preds)
plt.title('Data points in 3D PCA axis', fontsize=20)
plt.show()

# export the clustering result
r=pd.DataFrame(c_preds)
anime_reslut=pd.concat([anime["ID"],anime["Name"],anime["Genre"],anime["Studio"],anime["members"],anime["Rating"],r],axis=1)
anime_reslut.to_csv("/Users/xu/Desktop/re.csv")

anime['cluster'] = c_preds
print(anime.head(10))

# seperating the clusters
c0_data = anime[anime['cluster']==0]
c1_data = anime[anime['cluster']==1]
c2_data = anime[anime['cluster']==2]
c3_data = anime[anime['cluster']==3]

# functions
def count_word(df, ref_col, liste):
    keyword_count = dict()
    for s in liste: keyword_count[s] = 0
    for liste_keywords in df[ref_col].astype(str).str.split(','):
        if type(liste_keywords) == float and pd.isnull(liste_keywords): continue
        for s in [s for s in liste_keywords if s in liste]:
            if pd.notnull(s): keyword_count[s] += 1
    # convert the dictionary in a list to sort the keywords by frequency
    keyword_occurences = []
    for k,v in keyword_count.items():
        keyword_occurences.append([k,v])
    keyword_occurences.sort(key = lambda x:x[1], reverse = True)
    return keyword_occurences, keyword_count

from wordcloud import WordCloud
def makeCloud (Dict, name, color) :
    words = dict()
    for s in Dict:
        words[s[0]] = s[1]
        wordcloud1 = WordCloud(
                      width=1500,
                      height=500,
                      background_color=color,
                      max_words=20,
                      max_font_size=500,
                      normalize_plurals=False)
        wordcloud1.generate_from_frequencies(words)
    fig = plt.figure(figsize=(12, 8))
    plt.title(name)
    plt.imshow(wordcloud1)
    plt.axis('off')
    plt.show()

# plot wordclouds for Studio variable in each cluster
set_keywords = set()
for liste_keywords in c0_data['Studio'].astype(str).str.split(',').values:
    if isinstance(liste_keywords, float): continue  # only happen if liste_keywords = NaN
    set_keywords = set_keywords.union(liste_keywords)
keyword_occurences, dum = count_word(c0_data, 'Studio', set_keywords)
makeCloud(keyword_occurences[0:10], "cluster 1", "lemonchiffon")

set_keywords = set()
for liste_keywords in c1_data['Studio'].astype(str).str.split(',').values:
    if isinstance(liste_keywords, float): continue  # only happen if liste_keywords = NaN
    set_keywords = set_keywords.union(liste_keywords)
keyword_occurences, dum = count_word(c1_data, 'Studio', set_keywords)
makeCloud(keyword_occurences[0:10], "cluster 2", "white")

set_keywords = set()
for liste_keywords in c2_data['Studio'].astype(str).str.split(',').values:
    if isinstance(liste_keywords, float): continue  # only happen if liste_keywords = NaN
    set_keywords = set_keywords.union(liste_keywords)
keyword_occurences, dum = count_word(c2_data, 'Studio', set_keywords)
makeCloud(keyword_occurences[0:10], "cluster 3", "snow")

set_keywords = set()
for liste_keywords in c3_data['Studio'].astype(str).str.split(',').values:
    if isinstance(liste_keywords, float): continue  # only happen if liste_keywords = NaN
    set_keywords = set_keywords.union(liste_keywords)
keyword_occurences, dum = count_word(c3_data, 'Studio', set_keywords)
makeCloud(keyword_occurences[0:10], "cluster 4", "black")


# plot wordclouds for Genre variable in each cluster
set_keywords = set()
for liste_keywords in c0_data['Genre'].astype(str).str.split(',').values:
    if isinstance(liste_keywords, float): continue  # only happen if liste_keywords = NaN
    set_keywords = set_keywords.union(liste_keywords)
keyword_occurences, dum = count_word(c0_data, 'Genre', set_keywords)
makeCloud(keyword_occurences[0:10], "cluster 1", "lemonchiffon")


set_keywords = set()
for liste_keywords in c1_data['Genre'].astype(str).str.split(',').values:
    if isinstance(liste_keywords, float): continue  # only happen if liste_keywords = NaN
    set_keywords = set_keywords.union(liste_keywords)
keyword_occurences, dum = count_word(c1_data, 'Genre', set_keywords)
makeCloud(keyword_occurences[0:10], "cluster 2", "white")

set_keywords = set()
for liste_keywords in c2_data['Genre'].astype(str).str.split(',').values:
    if isinstance(liste_keywords, float): continue  # only happen if liste_keywords = NaN
    set_keywords = set_keywords.union(liste_keywords)
keyword_occurences, dum = count_word(c2_data, 'Genre', set_keywords)
makeCloud(keyword_occurences[0:10], "cluster 3", "snow")


set_keywords = set()
for liste_keywords in c3_data['Genre'].astype(str).str.split(',').values:
    if isinstance(liste_keywords, float): continue  # only happen if liste_keywords = NaN
    set_keywords = set_keywords.union(liste_keywords)
keyword_occurences, dum = count_word(c3_data, 'Genre', set_keywords)
makeCloud(keyword_occurences[0:10], "cluster 4", "black")
