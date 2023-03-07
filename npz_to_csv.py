
import numpy as np
import pandas as pd


data = np.load('data/labels.npz',allow_pickle=True)
for item in data.files:
    label = pd.DataFrame(data[item])
    print(data[item])



# a=pd.DataFrame(data=file)
# a.to_excel('data/labels.xlsx',index=None)
# a.to_csv('data/labels.csv',index=None)
