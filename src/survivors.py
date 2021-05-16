import torch
import numpy as np
import torch.nn as nn

'''

'''

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
match_list=torch.load('match_list.pt').to(device)
matches=np.load('matches.npy')

def surviving_matches(thresholds):
    A=torch.index_select(thresholds, 1, match_list[:,0].int())
    B=match_list[:,1]
    C=torch.index_select(thresholds, 1, match_list[:,2].int())
    D=match_list[:,3]
    return torch.sum(torch.logical_and(torch.gt(A ,B), torch.gt(C ,D)), dim=1)
def relative_surviving_matches(thresholds):
    return torch.div(surviving_matches(thresholds), len(match_list)).reshape(-1,1)

def evalues_per_pair(pair):
    set1=torch.logical_and(torch.eq(match_list[:,0], relevant_matches[:,pair][1]), torch.eq(match_list[:,2], relevant_matches[:,pair][0]))
    set2=torch.logical_and(torch.eq(match_list[:,0], relevant_matches[:,pair][0]), torch.eq(match_list[:,2], relevant_matches[:,pair][1]))
    ev1_set1=match_list[set1,1]
    ev0_set1=match_list[set1,3]
    ev0_set2=match_list[set2,1]
    ev1_set2=match_list[set2,3]
    ev1=torch.cat([ev1_set1, ev1_set2])
    ev0=torch.cat([ev0_set1, ev0_set2])
    return torch.stack([ev0, ev1], dim=1)

def pair_surviving_matches(pair, thresholds):
    ev_tensor=pair_evalues[pair]
    p1=torch.lt(ev_tensor[:,0], thresholds[:,0].reshape(-1,1))
    p2=torch.lt(ev_tensor[:,1], thresholds[:,1].reshape(-1,1))
    return torch.div(torch.sum(torch.logical_and(p1,p2), dim=1), weights[pair]).float()

minimum_matches=10
where=np.where(matches>minimum_matches)
relevant_matches=torch.tensor(where).to(device)
weights=torch.tensor(matches[where])
indices=torch.gt(relevant_matches[1], relevant_matches[0])
relevant_matches=relevant_matches[:,indices]
weights=weights[indices].to(device)

pair_evalues=[]
for i in range(len(weights)):
    print(i, end='\r')
    pair_evalues.append(evalues_per_pair(i))

class pair_approximator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 8),
            nn.LeakyReLU(),
            nn.Linear(8,4),
            nn.LeakyReLU(),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        input=torch.log10(x)
        return torch.sigmoid(self.layers(input))
torch.manual_seed(0)
batch_size=32
threshold_on_vertex=0.9
loss_threshold=0.001
pair_matches_approximator=[pair_approximator().to(device) for i in range(len(weights))]
optimizer_list = [torch.optim.AdamW(net.parameters(), lr=0.1) for net in pair_matches_approximator]
loss=torch.nn.MSELoss()
losses=[]
toprocess=np.arange(len(weights))
epoch=0
while len(toprocess)>0:
    epoch+=1
    randomint=torch.randint(high=10, size=(batch_size, 2)).float()
    x=(torch.rand(batch_size, 2)*(10**-randomint)).reshape(batch_size, 2).to(device)
    total_loss=0
    for i in toprocess:
        l_hat=pair_matches_approximator[i](x)
        l_true=pair_surviving_matches(i, x).reshape(-1,1)
        l=loss(l_hat, l_true)
        if l<loss_threshold and pair_matches_approximator[i](torch.ones(1,2).to(device))>threshold_on_vertex:
            torch.save(pair_matches_approximator[i].state_dict(), "pair_models/pair%d.pt"%i)
            toprocess=np.delete(toprocess, np.where(toprocess==i))
        total_loss+=l.item()
        optimizer_list[i].zero_grad()
        l.backward()
        optimizer_list[i].step()
    losses.append(total_loss)
    print("epoch: "+str(epoch)+", loss: "+"{:.6f}".format(total_loss)+", pairs to process: "+str(len(toprocess)), end='\r')
torch.save(weights, "weights.pt")
torch.save(relevant_matches, "relevant_matches.pt")
