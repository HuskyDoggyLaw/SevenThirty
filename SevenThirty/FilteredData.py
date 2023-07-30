import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as pt

# Read the data only the columns: created_year, country_rank and Country
df = pd.read_excel('C:/Users/manne/source/repos/PythonApplication1/PythonApplication1/data/youtube.xlsx', usecols=["created_year", "country_rank", "Country"])

# Find only the countries: US or India or Japan or blanks
df = df.query('Country =="United States" or Country =="India" or Country =="Japan" or Country =="South Korea" or Country.isna()')

# Fill all the blanks with India
df['Country'].fillna('India')

# find the outlier which is less than 5% and update all those values to 2020
perc = np.percentile(df['created_year'], 5)
df.loc[df['created_year'] <= perc, 'created_year'] = 2020

# Show the scattered plot
sb.scatterplot(x='Country', y='created_year', data=df)
pt.show()


