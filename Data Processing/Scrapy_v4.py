# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 22:05:46 2019

@author: chest
"""

from scrapy.selector import Selector
from scrapy.http import HtmlResponse
from time import sleep, time
from os import chdir
from random import randrange
import requests
import numpy as np
import re
import pandas as pd

#set directory
chdir("C:\\Users\\chest\\Desktop\MTech\\Recommender System\\Anime Project\\Anime_files")

#functions
def createUrl(ID):
    base = "https://myanimelist.net/anime/"
    total_url = []
    for id in ID:
        url = base + str(id)
        total_url.append(url)
    return total_url

def flatlist(li):
    a = "".join(map(str, li))
    return(a)

def checklist(li, ind):
    if len(li) == 0:
        return('None')
    else:
        return(li[ind])

def basic_scrape(id_list):
    url_list = createUrl(id_list)
    [missingStatus, missingStudio, missingDuration, missingRating] = [[],[],[],[]]
    [missingSynopsis, missingImg] = [[],[]]
    [missingGenre, missingType, missingScore] = [[],[],[]]
    typ, stat, sco = (1,3,1)
    base1 = 2
    base2 = 3
    original = np.array([7,9,10,11])
    start = time()
    for ind, url in enumerate(url_list): 
        r = requests.get(url)
        sel = Selector(text = r.content)
        #main check for website exist
        check = str(sel.xpath("//div[@class='error404']/@title").extract())
        sleep(1)
        if check == "[]":
            #check which option to use (1st layer check)
            h1 = str(sel.xpath("//div[@class='js-scrollfix-bottom']/h2[1]/text()").extract())
            if "Alternative" in h1:
                B1 = base1
                B2 = base2
            else:
                B1 = base1 - 1
                B2 = base2 - 1
            base_xpath = f"//div[@class='js-scrollfix-bottom']/h2[{B1}]/following-sibling::div"
            base_xpath2 = f"//div[@class='js-scrollfix-bottom']/h2[{B2}]/following-sibling::div"
            #Perform loop reading based on length for unstable variables
            length = len(sel.xpath(base_xpath))
            missingSynopsis.append(str(sel.xpath("//div[@id='content']/table//td/span[@itemprop='description']/text()").extract()))
            missingImg.append(str(sel.xpath("//div[@class='js-scrollfix-bottom']/div[@style='text-align: center;']//img/@src").extract()))
            missingType.append(checklist(sel.xpath(f'{base_xpath}[{typ}]//text()').extract(),-1))
            missingStatus.append(checklist(sel.xpath(f'{base_xpath}[{stat}]//text()').extract(),-1))
            missingScore.append(checklist(sel.xpath(f'{base_xpath2}[{sco}]//text()').extract(),3))
            for div in range(1,length+1):
                span = sel.xpath(f'{base_xpath}[{div}]/span/text()').extract()
                if span == ["Studios:"]:
                    sel_studio = sel.xpath(f'{base_xpath}[{div}]//text()').extract()
                    sel_studio = flatlist(sel_studio[2:len(sel_studio)])
                    continue
                if span == ["Genres:"]:
                    sel_gen = sel.xpath(f'{base_xpath}[{div}]//text()').extract()
                    sel_gen = flatlist(sel_gen[2:len(sel_gen)])
                    continue
                if span == ["Duration:"]:
                    sel_dur = sel.xpath(f'{base_xpath}[{div}]//text()').extract()
                    sel_dur = checklist(sel_dur, -1)
                    continue
                if span == ["Rating:"]:
                    sel_rate = sel.xpath(f'{base_xpath}[{div}]//text()').extract()
                    sel_rate = checklist(sel_rate, -1)
                    continue
            missingStudio.append(sel_studio)
            missingGenre.append(sel_gen)
            missingDuration.append(sel_dur)
            missingRating.append(sel_rate)
            print(ind)
            sleep(1)
        if "This page doesn't exist" in str(check):
            missingStatus.append("No Page")
            missingStudio.append("No Page")
            missingDuration.append("No Page")
            missingRating.append("No Page")
            missingSynopsis.append("No Page")
            missingImg.append("No Page")
            missingGenre.append("No Page")
            missingType.append("No Page")
            missingScore.append("No Page")
    end = time()
    print(end - start)
    df_data = {'ID':list(id_list), 'Status':missingStatus, 'Type':missingType, 'Studio':missingStudio, 
             'Genre':missingGenre, 'Duration':missingDuration, 'Rating':missingScore, 'Category':missingRating,
             'Synopsis':missingSynopsis,'Image':missingImg}
    df = pd.DataFrame(df_data)    
    return(df)

def getIndex(mainDf,mainCol, missingIndex, replaceDf):
    '''get index of missing in main Df based on replaceDf
    column should be the unique primary key'''
    output = []
    for i in mainDf.loc[missingIndex, mainCol]:
        ind = replaceDf.index[replaceDf.ID == i]
        output.append(ind)
    return(output)

#there are problematic pages that requries more robust check 
problem_df = pd.read_excel("problem_ids.xlsx", index_col=None, header=0)
problem_df.columns = ["ID", "Name"]
correction = basic_scrape(problem_df.iloc[:,0].values)
correction = pd.merge(problem_df, correction, on="ID")
correction = correction.loc[:,['ID', 'Name', 'Genre', 'Type', 'Status', 'Studio', 'Duration', 'Rating',
       'Category', 'Synopsis', 'Image']]

#to rectify the problem of wrong columns
raw_scrape = pd.read_excel("scraped_raw.xlsx", index_col=0)
raw_scrape = raw_scrape.set_index(raw_scrape.ID)
raw_scrape = raw_scrape.drop(problem_df.iloc[:,0].values, axis=0)
raw_scrape = pd.concat([raw_scrape, correction])
raw_scrape.to_excel("raw_scrape.xlsx")







