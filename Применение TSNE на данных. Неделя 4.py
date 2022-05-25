import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 5)
import os
#os.environ["PROJ_LIB"] = "C:\\Python\\Anaconda\\pkgs\\proj-8.0.1-h1cfcee9_0\\Library\share\proj"

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

digit = load_digits()
img = digit.images
#print('размерность img: ',img.shape)
#plt.imshow(img[0], cmap=plt.cm.Greys_r)
#plt.show()
X = img.reshape(-1, 64) # 64 = 8x8, -1 = все остальные размерности в первую размерность.
#print('размерность X: ', X.shape)
tsne = TSNE(n_components=2, perplexity=30.)
tsne.fit(X)
Z = tsne.embedding_
print('размерность Z: ', Z.shape)
y = digit.target
#plt.scatter(Z[:, 0], Z[:,1], c=y) #одинаковые рисунки цифр близки по параметрам
#plt.show()
permplex = [1, 3, 5, 10, 30, 50]

for p in permplex:
    tsne = TSNE (n_components=2, perplexity=p)
    tsne.fit(X)
    Z = tsne.embedding_
    fit, ax = plt.subplots(1, 1)
    ax.scatter(Z[:,0], Z[:, 1], c=y)
    ax.set_title('Perplexity {}'.format(p))
    plt.show() #Перплексия регулирует упор на глобальную или локальную структуру
    # при ее увеличении - уменьшение количества всё больших кластеров
