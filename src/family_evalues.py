import pandas as pd
import numpy as np
import torch

'''
This script produces the tensor 'family_evalues', which stores for each family the list of evalues of its hits.
'''

df=pd.read_csv("data.csv")
families=np.load("families.npy", allow_pickle=True)
family_sizes=np.load("sizes.npy")
data=df.to_numpy()
matches=np.zeros((len(families), len(families)))
ev_tensor=np.ones((len(families), len(families), 2))
family_index={family: np.where(families==family)[0][0] for family in families}
family_evalues=np.zeros((len(families), np.max(family_sizes)))
index=np.zeros(len(families), dtype=np.int)
for i in range(len(data)):
    print(i, "/", len(data), end='\r')
    family=family_index[data[i,1]]
    evalue=data[i,2]
    family_evalues[family, index[family]]=evalue
    index[family]+=1
family_evalues=torch.tensor(family_evalues, requires_grad=True)
family_sizes=torch.tensor(family_sizes, dtype=torch.float, requires_grad=True)
torch.save(family_evalues, 'family_evalues.pt')
torch.save(family_sizes, 'family_sizes.pt')
