import pandas as pd
import numpy as np
import torch

'''
In this script matches are identified and stored in a matrix 'matches' (whose i,j-th entry is the number of matches between
families i and j) and in a tensor 'match_list' (for each match, it stores the names of the two involved families and the
respective e-values).
'''

def intersection_size(start1, start2, end1, end2):
    return len(range(max(start1, start2), min(end1, end2)+1))
def union_size(start1, start2, end1, end2):
    return len(range(start1, end1+1))+len(range(start2, end2+1))-intersection_size(start1, start2, end1, end2)
def match(row1, row2):
    start1=int(row1[3])
    start2=int(row2[3])
    end1=int(row1[4])
    end2=int(row2[4])
    if intersection_size(start1, start2, end1, end2)/union_size(start1, start2, end1, end2)>0.8:
        return True
    return False

df=pd.read_csv("data.csv")
families=df.family.unique()
matches=np.zeros((len(families), len(families)))
ev_tensor=np.ones((len(families), len(families), 2))
# Sorting the dataframe by protein name allows a faster loop over data points
df=df.sort_values("protein", ignore_index=True)
family_index={family: np.where(families==family)[0][0] for family in families}
family_sizes=np.zeros_like(families, dtype=np.int32)
i=0
n=len(df)
data=df.to_numpy()
match_list=[]
for i in range(n-1):
    print(str(i)+'/'+str(n), end='\r')
    ith_row=data[i]
    protein=ith_row[0]
    family1=ith_row[1]
    evalue1=float(ith_row[2])
    family_sizes[family_index[family1]]+=1
    j=i+1
    jth_row=data[j]
    while jth_row[0]==protein:
        family2=jth_row[1]
        evalue2=float(jth_row[2])
        if family2!=family1 and match(ith_row, jth_row):
            fam1index=family_index[family1]
            fam2index=family_index[family2]
            matches[fam1index, fam2index]+=1
            match_list.append(np.array([fam1index, evalue1, fam2index, evalue2]))
        j+=1
        if j==n:
            family_sizes[family_index[family2]]+=1
            break
        jth_row=data[j]
# Matrix 'matches' has to be made symmetrical
for i in range(len(families)):
    for j in range(i, len(families)):
        matches[i,j]+=matches[j,i]
        matches[j,i]=matches[i,j]
np.save("sizes", family_sizes)
np.save("matches", matches)
np.save("families", families)
match_list=torch.tensor(match_list)
torch.save(match_list, 'match_list.pt')
