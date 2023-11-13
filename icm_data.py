import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Create encoder
encoding = LabelEncoder()

# Create pandas dataframe
df = pd.read_csv("metadata.csv")
df = df.drop(['folder', 'filename', 'embryo_ivf_id', 'embryo_guid', 'stage', 'guid', 'thawed'],
             axis=1)  # drop unnecessary data
df = df.reset_index(drop=True)
print(df['grade'].value_counts())  # print the number of samples per class

df = df[df["grade"].str.contains("2") == False]
df = df.reset_index(drop=True)

icm = df
icm['grade'] = icm['grade'].replace(
    {'AB': 'A', 'AA': 'A', 'BA': 'B', 'AC': 'A', 'CA': 'C', 'BB': 'B', 'BC': 'B', 'CC':
        'C', 'CB': 'C', 'DD': 'D'})
print(icm['grade'].value_counts())

icm['grade'] = encoding.fit_transform(icm['grade'])
icm.to_csv('icm_four_classes.csv')  # export csv file for ICM classification
