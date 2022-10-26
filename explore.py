import pandas as pd 
import os
csv_path = os.getcwd()+"/covid_articles_raw.csv"
df = pd.read_csv(csv_path, encoding='utf-8')

df[:25].to_csv(csv_path.replace('.csv','_first_250_.csv'), index=False)

# category_value_counts = df['category'].value_counts()
# category_value_counts.to_csv(csv_path.replace('.csv','_category_value_counts.csv'), encoding='utf-8')

