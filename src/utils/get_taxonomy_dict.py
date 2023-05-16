import pickle

import pandas as pd
from sklearn.preprocessing import LabelEncoder

taxonomy = pd.read_csv("../data/eBird_Taxonomy_v2021.csv")
print(taxonomy.columns)

#ORDER1, FAMILYを分類として使用
#Nanは最後のクラスとして符号化される
order_le = LabelEncoder()
taxonomy['ORDER1_encoded'] = order_le.fit_transform(taxonomy['ORDER1'])
family_le = LabelEncoder()
taxonomy['FAMILY_encoded'] = family_le.fit_transform(taxonomy['FAMILY'])

#SPECIES_CODEと[ORDER1, FAMILY]の組を辞書化
taxonomy_dict = {}

for index, row in taxonomy.iterrows():
    key = row["SPECIES_CODE"]
    value = [row["ORDER1_encoded"], row['FAMILY_encoded']]
    taxonomy_dict[key] = value

with open("../csv/taxonomy_dict.pkl", "wb") as tf:
    pickle.dump(taxonomy_dict, tf)