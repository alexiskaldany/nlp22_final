import pandas as pd 
import os
csv_path = os.getcwd()+"/covid_articles_raw.csv"
df = pd.read_csv(csv_path, encoding='utf-8')

df[:25].to_json(csv_path.replace('.csv','.json'),orient='records')