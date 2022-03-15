from matplotlib import pyplot as plt
import pandas as pd
from numpy.random import exponential

#df = pd.DataFrame({'x':range(20), 'y':exponential(10, 20)})
#df.y.hist()
data1 = exponential(5, 20)
data2 = exponential(6, 20)
plt.plot(data1, label='Первый игрок')
plt.scatter(range(len(data2)), data2, color='red', label="Второй игрок")
plt.title('Результаты', fontdict={'fontsize':20, 'color':'green'})
plt.xlabel('номер попытки')
plt.ylabel('Результат')
plt.legend()
plt.xticks(range(0, 20, 4))
#plt.savefig('results.png')
plt.show()

