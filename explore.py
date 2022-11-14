import pandas as pd 
import os
import seaborn as sns

csv_path = os.getcwd()+"/covid_articles_raw.csv"
# df = pd.read_csv(csv_path, encoding='utf-8')

#df =pd.read_csv(csv_path.replace('.csv','_first_250_.csv'), encoding='utf-8')

# category_value_counts = df['category'].value_counts()
# category_value_counts.to_csv(csv_path.replace('.csv','_category_value_counts.csv'), encoding='utf-8')

# df["length"] = df["content"].apply(lambda x: len(str(x)))
# # print(df["length"].describe())
# value_counts = df["length"].value_counts()
# value_counts.to_csv(csv_path.replace('.csv','_length_value_counts.csv'), encoding='utf-8')
# value_counts.plot(kind='bar')

# df = df[["length", "category"]]
# df.to_csv(csv_path.replace('.csv','_length_category.csv'), encoding='utf-8')

length_cat = pd.read_csv(csv_path.replace('.csv','_length_category.csv'), encoding='utf-8')

import seaborn as sns
sns.set(style="darkgrid")
# ax = sns.countplot(x="length", hue="category", data=length_cat)
# ax=sns.violinplot(x="category", y="length", data=length_cat)
# ax.figure.savefig(csv_path.replace('.csv','_length_category.png'))
dist = sns.displot(length_cat["length"], kde=True)
dist.figure.savefig(csv_path.replace('.csv','_length_category_dist.png'))

