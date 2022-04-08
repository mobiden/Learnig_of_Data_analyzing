import plotly.offline as offline
from plotly.graph_objs import *
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
import plotly.io as pio
pio.renderers.default = "chrome"
offline.init_notebook_mode()


df = pd.read_csv('https://raw.githubusercontent.com/yankev/test/master/life-expectancy-per-GDP-2007.csv')
df.sort_values('gdp_percap', inplace=True)
#trace = Scatter(x=df.gdp_percap,
#                y=df.life_exp
 #               )
df['population'] = df.country.str.split(':').apply(lambda words: float(words[-1]))
df['name'] = df.country.str.split(':').apply(lambda words: words[1].split('<br>')[0])

americas = df[(df.continent== "Americas")]
europe = df[(df.continent== "Europe")]
trace1 = Scatter(x=americas.gdp_percap,
                 y=americas.life_exp,
                 name='Americas',
                 mode='markers', #тип графика
                 text=americas.name, #подписи точек
          #       marker={'size': americas.population/americas.population.max()*20} #pазмер точек
                 )
trace2=  Scatter(x=europe.gdp_percap,
                 y=europe.life_exp,
                 name='Europe',
                 mode='markers',
                 text=europe.name,
                 )
data = [trace1, trace2]
offline.plot(data)

