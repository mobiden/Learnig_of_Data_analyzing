import pandas as pd
from matplotlib import pyplot as plt
df = pd.read_csv('titanic.csv')
df.head
plt.plot(df.Fare.plot.hist())
plt.show()