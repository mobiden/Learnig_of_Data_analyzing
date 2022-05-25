import pandas as pd
from IPython.display import Image

calendar = pd.read_csv('Data/calendar.csv', index_col=0)
reviews = pd.read_csv('reviews.csv')
listings = pd.read_csv('listings.csv', index_col=0)

# print(listings.head(2))
table_merge = pd.merge(listings, reviews,
               left_on=['id'], right_on=['listing_id'], how='right', indicator=True
               )
calendar.set_index('listing_id', inplace=True)
reviews.set_index('listing_id', inplace=True)
calendar.join(reviews, lsuffix='listing_id', rsuffix='listing_id')
