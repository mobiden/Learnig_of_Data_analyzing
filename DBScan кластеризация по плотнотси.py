import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12,5)
import os
os.environ["PROJ_LIB"] = "C:\\Python\\Anaconda\\pkgs\\proj-8.0.1-h1cfcee9_0\\Library\share\proj"


#df_geo = pd.read_csv('geo_data.txt', sep='\t', header=None,
#                     names=['lat', 'long'])/10

df_geo = pd.read_csv('bog_clean2.csv')
#print(df_geo.head())


import mpl_toolkits.basemap as bm

def plot_geo(lat, long, labels= None):
    try:
        lllat, lllon = lat.min() - 1, long.max() + 1
        urlat, urlon = lat.max() + 1, long.min() - 1
        plt.figure(figsize=(10,10))

        m = bm.Basemap(
            llcrnrlon= lllon,
            llcrnrlat= lllat,
            urcrnrlon= urlon,
            urcrnrlat=urlat,
            projection='merc',
            resolution='h'
        )
        m.drawcoastlines(linewidth=0.5)
        m.drawmapboundary(fill_color='#47A4C9', zorder=1)
        parallels = np.linspace(lllat, urlat, 10)
        m.drawparallels(parallels, labels=[1,0,0,0], fontsize=10)
        meridians = np.linspace(urlon, lllon, 10)
        m.drawmeridians(meridians, labels=[0,0,0,1], fontsize=10)
        m.scatter(long, lat, latlon=True, cmap=plt.cm.jet,
                  zorder=3, lw=0, c=labels)
    except:
        print('Не так')
        plt.scatter(x=long, y=lat, c=labels, cmap=plt.cm.jet)
        plt.axis('equal')
#plot_geo(df_geo['pickup_longitude'].values, df_geo['pickup_latitude'].values)
#plt.show()

from sklearn.neighbors import NearestNeighbors
km_in_radian = 6371.0088
X = pd.DataFrame(df_geo[['pickup_longitude', 'pickup_latitude']])
X = np.radians(X)
#print(X.head())
model = NearestNeighbors(n_neighbors=20, algorithm='ball_tree', metric='haversine') # расчет ближайших
model.fit(X)
dist, _ = model.kneighbors(X, n_neighbors=20, return_distance=True) #дистанция до 20 ближайших точек
#print(dist.shape)
dist = dist[:, -1] #выбор последнего столбца
dist = np.sort(dist)
#plt.plot(dist) #выбор точки, где график начнет резко возрастать, для определения epsilon
#plt.show()

eps = 0.0002
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=20, metric='haversine', algorithm='ball_tree')
dbscan.fit(X)
labels = dbscan.labels_ #сохранение меток
#print(pd.Series(labels).value_counts())
idx = labels!=-1 #маска для удаления выбросов, True для невыбросов
plot_geo(df_geo.loc[idx, 'pickup_longitude'].values, df_geo.loc[idx, 'pickup_latitude'].values,
         labels=labels[idx])
plt.show()

