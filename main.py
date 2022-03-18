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
trace = Scatter(x=df.gdp_percap,
                y=df.life_exp
                )
#fig = px.scatter(x=df.gdp_percap,
#                y=df.life_exp
#                )

#fig.write_html("file.html")

data = [trace]
offline.plot(data)

