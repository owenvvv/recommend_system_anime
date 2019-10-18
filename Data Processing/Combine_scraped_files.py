# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 20:20:46 2019

@author: chest
"""
from os import chdir
import pandas as pd
#function to concatenate the different files into one dataframe

#set directory
chdir("C:\\Users\\chest\\Desktop\\MTech\\Recommender System\\Anime Project\\Anime_files")

#reading the file
anime = pd.read_csv("anime.csv")

#detecting missing values
anime.isnull().any()
genreIndex = pd.Series(anime.index[anime.genre.isnull()])
typeIndex = pd.Series(anime.index[anime.type.isnull()])
commonMissing = set(genreIndex).intersection(set(typeIndex))
totalMissing = pd.concat([genreIndex, typeIndex]).unique()

#create a subset of anime_df [anime_idf, name, type, genre)
anime_new = anime.drop(totalMissing, axis=0)
anime_new = anime_new.iloc[:,[0,1,2,3]]
anime_new.rename(columns = {"anime_id":"ID", "name":"Name",
                            "genre":"Genre", "type":"Type"}, inplace=True)

#create a list of file_names
name_list = ["anime_" + str(end) + ".xlsx" for end in range(100,12300,100)]
main_df = pd.DataFrame(columns= ['ID','Status', 'Type','Studio','Genre',
                                  'Duration','Rating', 'Category','Synopsis','Image'])

for name in name_list:
    file = pd.read_excel(name, index_col=0)
    main_df = pd.concat([main_df, file])

#to add on remaining 10 dataframe
remaining_data = pd.read_excel("anime_12210.xlsx", index_col=0)
main_df = pd.concat([main_df, remaining_data])

#to merge main_df into anime_new
anime_new = pd.merge(anime_new, main_df.iloc[:,[0,1,3,5,6,7,8,9]], on="ID")

#impute missing values (Just read if you have the file)
missingID = anime.iloc[totalMissing, :]["anime_id"].unique()
missing_df = basic_scrape(missingID)
genre_df = anime.iloc[genreIndex, [0,1,3]]
genre_df.insert(2, "genre", list(missing_df.loc[0:61, "Genre"]))
loc = getIndex(anime, "anime_id", commonMissing, missing_df)
for i, x in zip(list(commonMissing), loc):
    output = missing_df.iloc[x,2].values
    genre_df.loc[i,"type"] = output

type_df = anime.iloc[typeIndex, [0,1,2,3]]
loc2 = getIndex(anime, "anime_id", typeIndex, missing_df)
for i, x in zip(typeIndex, loc2):
    output2 = missing_df.iloc[x,2].values
    type_df.loc[i,"type"] = output2
type_df = type_df.drop(commonMissing, axis = 0)

cleanMiss = pd.concat([genre_df, type_df])
cleanMiss.rename(columns = {"anime_id":"ID", "name":"Name",
                            "genre":"Genre", "type":"Type"}, inplace=True)

cleanMiss = pd.merge(cleanMiss, missing_df.iloc[:,[0,1,3,5,6,7,8,9]], on="ID")

#Reading the imputted value file
cleanMiss = pd.read_excel("cleanMiss.xlsx", index_col=0)

