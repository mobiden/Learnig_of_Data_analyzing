import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 6)
import os
#os.environ["PROJ_LIB"] = "C:\\Python\\Anaconda\\pkgs\\proj-8.0.1-h1cfcee9_0\\Library\share\proj"
from tqdm import tqdm_notebook

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

filepath = 'user_ratedmovies.dat'
df_rates = pd.read_csv(filepath, sep='\t')

filepath = 'movies.dat'
df_movies = pd.read_csv(filepath, sep='\t', encoding='iso-8859-1')

from sklearn.preprocessing import LabelEncoder
#print(df_rates.head())
#print(df_movies.head())
#print(df_rates.userID.min(), df_rates.userID.max())
#print(df_rates.userID.nunique()) #количество уникальных пользователей

#перекодирование идентификаторов от "0" до количества уникальных записей
enc_user = LabelEncoder()
enc_movies = LabelEncoder()
enc_user = enc_user.fit(df_rates.userID.values)
enc_movies = enc_movies.fit(df_rates.movieID.values)

# удаление фильмов, которых нет в рейтингах
idx = df_movies.loc[:, 'id'].isin(df_rates.movieID)
df_movies = df_movies.loc[idx]

df_rates.loc[:, 'userID'] = enc_user.transform(df_rates.loc[:, 'userID'].values)
df_rates.loc[:, 'movieID'] = enc_movies.transform(df_rates.loc[:, 'movieID'].values)
df_movies.loc[:, 'id'] = enc_movies.transform(df_movies.loc[:, 'id'].values)

#print(df_rates.head())

# Матрица рейтингов (разреженные матрицы)
from scipy.sparse import coo_matrix, csr_matrix
R = coo_matrix((df_rates.rating.values, (df_rates.userID.values, df_rates.movieID.values)))

# SVD на матрице рейтингов

from scipy.sparse.linalg import svds

u, s, vt = svds(R, k=6)
#print('u, s, vt = ', u.shape, s.shape, vt.shape)

# поиск ближайших соседей для фильмов в сжатом признковом пространстве
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=10) #количество ближайших соседей - 10
v = vt.T # получаем обычную матрицу из транспонированной
nn.fit(v) #обучаем алгоритм
_, ind = nn.kneighbors(v, n_neighbors=10) #поиск индексов ближайших соседей, расстояния между ними не нужны
#print(ind[:10])

# названия ближайших фильмов
movies_titles = df_movies.sort_values('id').loc[:, 'title'].values

# создаем матрицу с ближайшими фильмами
cols = ['movie'] + ['nn_{}'.format(i) for i in range (1, 10)] # названия колонок
df_ind_nn = pd.DataFrame(data=movies_titles[ind], columns=cols)
#print(df_ind_nn.head())

# ищем для фильма Терминатор
idx = df_ind_nn.movie.str.contains('Terminator')
kk = df_ind_nn.loc[idx].head()
#print(df_ind_nn.loc[idx].head())


# Похожесть пользователей
from sklearn.metrics.pairwise import cosine_similarity as cosine_similarity

D = cosine_similarity(R) #метод считает косинусное расстояние по матрице без учета одинаковости фильмов
# print(D.shape())

from scipy.spatial.distance import  cosine, pdist, squareform
from sklearn.metrics import pairwise_distances

def similarity(u, v):
    # поиск пересечений между индексами у пользователей
    idx = (u != 0) & (v != 0)
    if np.any(idx): # если есть пересечение
        sim = -cosine(u[idx], v[idx]) + 1 # получаем из косинусного расстояния схожесть
        return sim
    else:
        return 0
# toarray - должна быть плотная матрица, pdist - подсчет косинусного расстояния
d = pdist(R.toarray(), metric=similarity)
#print(d.shape)
D = squareform(d) #собираем матрицу из большого вектора. матрица попарных косинусных похожестей между объектами
print (D.shape)



