import numpy as np
import pandas as pd

'''
This script reads the dataset and extract useful information for match checking and removal
(protein name, family name, evalue, starting and ending position in the alignment), stored in a .csv file
'''

target_names=[]
query_names=[]
e_values=[]
start=[]
end=[]
filename="MC_uniref_all.scanned"
with open(filename) as file:
    for i, line in enumerate(file):
        print(i, end='\r')
        line=line.split()
        try:
            int(line[0])
        except ValueError:
            continue
        target_names.append(line[0])
        query_names.append(line[3])
        e_values.append(line[6])
        start.append(line[17])
        end.append(line[18])

data={
    "protein": target_names,
    "family": query_names,
    "evalue": e_values,
    "start": start,
    "end":end
}
df=pd.DataFrame(data)
df.to_csv('data.csv', index=False)
