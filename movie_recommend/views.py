from django.shortcuts import render  # 渲染页面
from django.shortcuts import redirect
# from django.core.urlresolvers import reverse
from django.urls import reverse
import urllib
import random
from ast import literal_eval
import difflib
import Levenshtein

from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login  # 处理登录和退出
from django.contrib.auth import logout
from django.core.cache import cache
from movie_recommend.models import UserProfile, AnimeRated, AnimeData  # 数据模型
from movie_recommend.recommend_algos import *  # 推荐算法模块
from movie_recommend.load_data import *

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import os

BASE_DIR = os.path.dirname(os.path.abspath('__file__'))

def home(request):
    # cache.clear()
    context = {}

    if request.method == 'POST':
        post_data = request.POST
        data = {}
        data = post_data.get('data', None)
        if data:
            return redirect('%s?%s' % (reverse('home'),
                                       urllib.parse.urlencode({'q': data})))  # reverse逆向解析出url，重定向到查询过程，传参数：查询词data
        else:
            return render(request, 'movie_recommend/home.html', context)

    elif request.method == 'GET':

        get_data = request.GET
        data = get_data.get('q', None)
        titles = cache.get('cos_genre')

        if not titles:
            # load_anime()
            load_data()

            texts = []

        else:
            print('data loaded')

        if not data:
            return render(request, 'movie_recommend/home.html', context)

        if request.user.is_superuser:
            return render(request, 'movie_recommend/superusersignin.html', {})
        elif request.user.is_authenticated:
            userprofile = UserProfile.objects.get(user=request.user)
        else:
            return render(request, 'movie_recommend/pleasesignin.html', {})

        animeindex = AnimeData.objects.values_list('animeID', flat=True)
        animetitle = AnimeData.objects.values_list('title', flat=True)
        animeimage = AnimeData.objects.values_list('imagepath', flat=True)
        idandtitle = list(zip(animeindex, animetitle, animeimage))
        idsort = pd.DataFrame(idandtitle)

        def get_equal_rate(str1, str2):
            return Levenshtein.jaro_winkler(str1, str2)

        uinput = str(data)
        search = []
        for title in idsort[1]:
            search.append(get_equal_rate(title, uinput))
        idsort['search'] = search
        idsort.sort_values("search", inplace=True, ascending=False)
        idandtitle = list(zip(idsort[0][0:8], idsort[1][0:8], idsort[2][0:8]))
        context['animes'] = idandtitle
        return render(request, 'movie_recommend/query_results.html', context)


def auth(request):

    if request.method == 'GET':
        data = request.GET
        auth_method = data.get('auth_method')
        if auth_method == 'Sign In':
            return render(request, 'movie_recommend/signin.html', {})
        else:
            return render(request, 'movie_recommend/createuser.html', {})

    elif request.method == 'POST':
        post_data = request.POST
        name = post_data.get('name', None)
        pwd = post_data.get('pwd', None)
        pwd1 = post_data.get('pwd1', None)
        create = post_data.get('create', None)

        if name and pwd and create:

            if User.objects.filter(username=name).exists() or pwd != pwd1:
                return render(request, 'movie_recommend/userexistsorproblem.html', {})
            user = User.objects.create_user(username=name, password=pwd)
            uprofile = UserProfile()
            uprofile.user = user
            uprofile.name = user.username
            uprofile.save(create=True)

            user = authenticate(username=name, password=pwd)
            login(request, user)
            return render(request, 'movie_recommend/home.html', {})

        elif name and pwd:
            user = authenticate(username=name, password=pwd)
            if user:
                login(request, user)
                return render(request, 'movie_recommend/home.html', {})
            else:
                return render(request, 'movie_recommend/nopersonfound.html', {})


def signout(request):
    logout(request)
    return render(request, 'movie_recommend/home.html', {})

def rate_movie(request):

    data = request.POST.getlist('interest')
    rate = data
    userprofile = None

    if request.user.is_superuser:
        return render(request, 'movie_recommend/superusersignin.html', {})
    elif request.user.is_authenticated:
        userprofile = UserProfile.objects.get(user=request.user)
    else:
        return render(request, 'movie_recommend/pleasesignin.html', {})
    uid = userprofile.id
    for movieindx in rate:
        if AnimeRated.objects.filter(AnimeID=movieindx).filter(userid=uid).exists():
            mr = AnimeRated.objects.get(AnimeID=movieindx, userid=uid)
            mr.value = 5
            mr.save()
        else:
            mr = AnimeRated()
            mr.userid = uid
            mr.value = 5
            mr.AnimeID = movieindx
            mr.save()

    return render(request, 'movie_recommend/ratesuccess.html', {})

def movies_recs(request):
    userprofile = None
    if request.user.is_superuser:
        return render(request, 'movie_recommend/superusersignin.html', {})
    elif request.user.is_authenticated:
        userprofile = UserProfile.objects.get(user=request.user)
    else:
        return render(request, 'movie_recommend/pleasesignin.html', {})

    context = {}
    if not AnimeRated.objects.filter(userid=userprofile.id).exists():
        return render(request, 'movie_recommend/underminimum.html', context)
    reclist_all = AnimeRated.objects.filter(userid=userprofile.id).filter(value=5).values_list('AnimeID', flat=True)
    if len(reclist_all) == 1:
        return render(request, 'movie_recommend/underminimum.html', context)
    idandtitles = []
    likeidandtitles = []
    for aid in random.sample(list(reclist_all), 2):
        animeindex = AnimeData.objects.get(animeID=aid).id
        likename = AnimeData.objects.get(animeID=aid).title
        likeinamepath = AnimeData.objects.get(animeID=aid).imagepath
        animerec = rec_anime()
        finalindex = animerec.rec_II(animeindex)
        idlist = []
        for index in finalindex:
            idlist.append(index[0])
        animelist = AnimeData.objects.filter(id__in=idlist)
        likeidandtitle = {'likename': likename, 'likeinamepath': likeinamepath}
        idandtitle = []
        for animeone in animelist:
            x = (animeone.animeID, animeone.title, animeone.imagepath)
            idandtitle.append(x)
        idandtitles.append(idandtitle)
        likeidandtitles.append(likeidandtitle)

    context['animelike1'] = likeidandtitles[0]
    context['animes1'] = idandtitles[0]
    context['animelike2'] = likeidandtitles[1]
    context['animes2'] = idandtitles[1]

    uid = userprofile.id
    animerec = rec_anime()
    rec_uu = animerec.rec_UU(uid)
    uuidandtitle = []
    if rec_uu:
        animelist = AnimeData.objects.filter(animeID__in=rec_uu)
        for animeone in animelist:
            x = (animeone.animeID, animeone.title, animeone.imagepath)
            uuidandtitle.append(x)

    context['animelikeuu'] = uuidandtitle

    return render(request, 'movie_recommend/recommendations.html', context)
