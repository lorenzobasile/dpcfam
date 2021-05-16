import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from models import loss_approximator, pair_approximator, lift

'''
Final (approximate) gradient descent to optimize DPCfam thresholds
'''

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def lost_data(threshold, family):
    lost_datapoints=torch.stack([torch.sum(torch.gt(family_evalues[family,:].to(device), t)) for t in threshold])
    return torch.div(lost_datapoints, family_sizes[family]).reshape(-1, 1)

relevant_matches=torch.load("relevant_matches.pt")
weights=torch.load("weights.pt").to(device)
family_sizes=torch.load("family_sizes.pt").to(device)
families=np.load("families.npy", allow_pickle=True)
df=pd.read_csv("data.csv")
match_list=torch.load("match_list.pt")
family_evalues=torch.load("family_evalues.pt")
familiestoprocess=np.arange(len(families))
def surviving_matches(thresholds):
    A=torch.index_select(thresholds, 1, match_list[:,0].to(device).int())
    B=match_list[:,1].to(device)
    C=torch.index_select(thresholds, 1, match_list[:,2].to(device).int())
    D=match_list[:,3].to(device)
    return torch.sum(torch.logical_and(torch.gt(A ,B), torch.gt(C ,D)), dim=1)
def relative_surviving_matches(thresholds):
    return torch.div(surviving_matches(thresholds), len(match_list)).to(device).reshape(-1,1)



torch.manual_seed(0)
d=lift(len(families)).to(device)
try:
    d.load_state_dict(torch.load("lift_weights.pt"))
except FileNotFoundError:
    pass
family_loss_approximator=[loss_approximator().to(device) for i in range(len(families))]
pair_matches_approximator=[pair_approximator().to(device) for i in range(len(weights))]
for i in range(len(families)):
    print(i, end='\r')
    try:
        family_loss_approximator[i].load_state_dict(torch.load("ldata_models/loss%d.pt"%i, map_location=device))
        family_loss_approximator[i].eval()
    except FileNotFoundError:
        np.delete(familiestoprocess, i)
for i in range(len(weights)):
    print(i, end='\r')
    pair_matches_approximator[i].load_state_dict(torch.load("lmatches_models/pair%d.pt"%i))
    pair_matches_approximator[i].eval()
optimizer=torch.optim.AdamW(d.parameters(), lr=0.5)
scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.9999)
losses=[]
print("starting training")
alpha=0.8
beta=1-alpha
for epoch in range(1000):
    t=d().reshape(1,-1)
    l1=alpha*sum([pair_matches_approximator[i](t[:,relevant_matches[:,i]])*weights[i] for i in range(len(weights))])/len(match_list)
    l2=beta*sum([family_loss_approximator[i](t[:,i])*family_sizes[i] for i in familiestoprocess])/len(df)
    real1=alpha*relative_surviving_matches(t)
    real2=beta*sum([lost_data(t[:,i], i)*family_sizes[i] for i in range(len(families))])/len(df)
    l=l1+l2
    print("\n\nEpoch:"+str(epoch))
    print("\nRemaining matches: "+str(l1.item())+" "+str(real1.item()))
    print("\nLost data: "+str(l2.item())+" "+str(real2.item()))
    print("\nApproximated loss: "+str(l.item()))
    print("\nReal loss: "+str((real1+real2).item()))
    optimizer.zero_grad()
    l.backward()
    optimizer.step()
    scheduler.step()
    losses.append(l)
    torch.save(d.state_dict(), "lift_weights.pt")
torch.save(d(), "thresholds.pt")
print("Surviving matches: ", relative_surviving_matches(d().reshape(1,-1)).item())
print("Lost data: ", (sum([lost_data(d()[i].reshape(1), i)*family_sizes[i] for i in range(len(families))])/len(df)).item())
