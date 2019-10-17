from scipy.stats import pearsonr
from scipy.spatial.distance import cosine

import numpy as np
import unicodedata
import copy
import math
from django.core.cache import cache


class rec_anime(object):

    def __init__(self):
        self.cos_gerne = cache.get('cos_gerne')
        self.cos_dtm = cache.get('cos_dtm')
        self.uu_matrix = cache.get('uu_matrix')
        self.cluster = cache.get('cluster')

    def rec_II(self, animeid):
        cos_dtm = self.cos_dtm
        cos_gerne = self.cos_gerne
        c = 0.2  # C  is the ratio of differernt models
        cosd = cos_dtm.applymap(lambda x: x * c)
        cos_final = cos_gerne.add(cosd, fill_value=0)  # Hybrid Matrix

        # Find the indexlist with the same or different cluster
        cluster = self.cluster
        cluster1 = cluster.loc[animeid].cluster
        cluster_same = cluster[(cluster['cluster'] == cluster1)]
        cluster_same = cluster_same['Index']
        cluster_diff = cluster[(cluster['cluster'] != cluster1)]
        cluster_diff = cluster_diff['Index']
        # Split the similarity matrix
        cos_same = cos_final.loc[cluster_same]
        cos_diff = cos_final.loc[cluster_diff]
        # Calculate the top4(except itself) of same cluster
        similar_animes_same = list(enumerate(cos_same[str(animeid)]))
        final_list_same = sorted(similar_animes_same, key=lambda x: x[1], reverse=True)[1:5]
        # Calculate the top4(except itself) of different cluster
        similar_animes_diff = list(enumerate(cos_diff[str(animeid)]))
        final_list_diff = sorted(similar_animes_diff, key=lambda x: x[1], reverse=True)[0:2]
        # Conbine two lists
        final_list = final_list_same + final_list_diff
        return final_list

    def rec_UU(self, uid):
        uu_matrix = self.uu_matrix
        try:
            uurec = list(uu_matrix.loc[uid])
        except KeyError:
            return None
        else:
            return uurec[0:6]
