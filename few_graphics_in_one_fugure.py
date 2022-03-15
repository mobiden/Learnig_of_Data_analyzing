from matplotlib import pyplot as plt

from numpy.random import exponential

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
for row, row_axes in enumerate(axes):
    for column, ax in enumerate(row_axes):
        ax.plot(exponential(column, 20))
        ax.set_title('Холст: колонка {} строка{}'.format(column+1, row+1))
fig.tight_layout() # распределение, чтобы все помещалось

#fig.savefig('allresults.png')
#plt.savefig('results.png')
plt.show()

