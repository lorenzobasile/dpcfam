import torch
import numpy as np
import torch.nn as nn
import os
from models import loss_approximator
from data import make_dataloader

'''
Training of neural approximators for L_data

cuda is not used in this file since when using a large dataset the tensor family_evalues becomes extremely large and
impossible to fit in a GPU memory
'''

try:
    os.mkdir("ldata_models")
except OSError:
    pass

batch_size=32

family_evalues=torch.load('family_evalues.pt')
family_sizes=torch.load('family_sizes.pt')
families=np.load('families.npy', allow_pickle=True)
#evalues=torch.load('all_evalues.pt')
#dataloader=make_dataloader(evalues, batch_size)
#print(len(dataloader.dataset))
def lost_data(threshold, family):
    lost_datapoints=torch.stack([torch.sum(torch.gt(family_evalues[family,:], t)) for t in threshold])
    return torch.div(lost_datapoints, family_sizes[family]).reshape(-1, 1)

torch.manual_seed(0)

family_loss_approximator=[loss_approximator() for i in range(len(families))]
optimizer_list = [torch.optim.AdamW(net.parameters(), lr=0.01) for net in family_loss_approximator]
loss=torch.nn.MSELoss()
losses=[]
toprocess=np.arange(len(families))
loss_threshold=0.0001
threshold_at_one=0.05
epoch=0
while len(toprocess)>0:
    epoch+=1
    #for x in dataloader:
        #x=x.reshape(batch_size,1)

    randomint=torch.randint(high=20, size=(batch_size,)).float()
    x=(torch.rand(batch_size)*(10**-randomint)).reshape(batch_size, 1)
    total_loss=0
    for i in toprocess:
        l_hat=family_loss_approximator[i](x)
        l_true=lost_data(x, i)
        l=loss(l_hat, l_true)
        total_loss+=l.item()
        optimizer_list[i].zero_grad()
        l.backward()
        if l<loss_threshold and family_loss_approximator[i](torch.ones(1,1))<threshold_at_one:
            torch.save(family_loss_approximator[i].state_dict(), "ldata_models/loss%d.pt"%i)
            toprocess=np.delete(toprocess, np.where(toprocess==i))
        optimizer_list[i].step()
    losses.append(total_loss)
    print("epoch: "+str(epoch)+", loss: "+"{:.6f}".format(total_loss)+", families to process: "+str(len(toprocess)), end='\r')
