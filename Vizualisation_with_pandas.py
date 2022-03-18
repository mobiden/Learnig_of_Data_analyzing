import pandas as pd
from matplotlib import pyplot as plt
import scipy
df = pd.read_csv('titanic.csv')
#df.Fare.plot.hist(bins=20)
#df.Fare.plot.kde()
#df.plot.scatter(x='Fare', y="Survived")

#df.groupby('Survived').Fare.plot.kde()
#plt.legend()
#plt.xlim(0,200)
#plt.show()

#ax=df.Fare.plot.hist()
#ax.set_title('Визуализация')
#plt.show()

fig, ax = plt.subplots(figsize=(10,5))
df.Survived.plot.kde(label='All', ax=ax)
for label, class_df in df.groupby('Pclass'):
    print(label)
    class_df.Survived.plot.kde(ax=ax, label=label)
plt.legend()
plt.title("Выживание в зависимости от класса каюты ")
plt.show()
