from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import numpy as np
import os
import json, pickle, time

# dataset for classification
class predictionDataset(Dataset):
    def __init__(self, notes, notes_dict, sequence_length=100):
        self.inputs = []
        self.outputs = []
        self.sequence_length = sequence_length
        self.num_notes = len(notes_dict)
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            self.inputs.append([notes_dict[char] for char in sequence_in])
            self.outputs.append(notes_dict[sequence_out])
        self.inputs = np.array(self.inputs).reshape(-1, self.sequence_length,1)/self.num_notes
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        
        return torch.FloatTensor(self.inputs[idx]),\
               torch.tensor(self.outputs[idx], dtype=torch.long)
    
data_dir = 'small_dataset'
notes_dict = json.load(open(os.path.join(data_dir, 'note_ids.json'), 'rb'))
num_notes = len(notes_dict)
artist = 'bach'
notes_path = os.path.join(data_dir, artist, 'train_notes')
with open(notes_path, 'rb') as file:
    notes = pickle.load(file)



class melodyNet(nn.Module):
    def __init__(self, obs_dim=1, hidden_dim=512, out_dim=525, layers=1):
        super(melodyNet, self).__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(self.obs_dim, hidden_dim, layers) 
        self.fc = nn.Linear(hidden_dim, out_dim)
        
    def forward(self,x):  # x in shape [batch_size, seq_len, obs_dim]
        # reshape,feed to lstm
        out = x.transpose(0,1)                           # reshape for lstm [seq_len, batch_size, inp_dim]
        out, _ = self.lstm(out)                       # [seq_len, batch_size, hidden_dim]            
        out = out[-1]         # [batch_size, hidden_dim]
        out = self.fc(out) # batch_size, out_dim
        return(out)


val_frac = 0.8
batch_size = 20
device = 'cuda:0'
myDataset = predictionDataset(notes, notes_dict)
dataset_size = len(myDataset)
val_size = int(val_frac*dataset_size)
train_size = dataset_size - val_size
train_dataset, val_dataset = random_split(myDataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=10)


myNet = melodyNet()

# optimizer
criterion = nn.CrossEntropyLoss()
epochs = 1000
learning_rate = 1e-3
optimizer = torch.optim.Adam(myNet.parameters(), lr=learning_rate)
lr_func = lambda e: 0.99**e
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
epochs = 100

myNet = myNet.to(device)
for x,y in train_loader:
    
    with torch.no_grad():
        out = myNet(x.to(device))
        loss = criterion(out, y.to(device))
        print(loss)
    break
    
    
# train
print('starting')
myNet = myNet.to(device)
t_start = time.time()
# tensorboard
run_path = 'logs/bach_seq2seq'
writer = SummaryWriter(os.path.join(run_path, 'tensorboard'))
os.system('cp train_seq2seq.py {}'.format(run_path))
def validate():
    val_loss_epoch, c = 0, 0
    for X,Y in val_loader:
        with torch.no_grad():
            outputs = myNet(X.to(device))
            loss = criterion(outputs.to(device), Y.to(device))
            val_loss_epoch += loss.data
            c += 1
    val_loss_epoch /= c
    return val_loss_epoch
    
val_loss = validate()
print('Val loss {}'.format(val_loss))
writer.add_scalar('validation loss', val_loss, 0)

best_val = val_loss

for e in range(epochs):
    loss_epoch, c = 0, 0
    for X,Y in train_loader:
        optimizer.zero_grad()
        outputs = myNet(X.to(device))
        loss = criterion(outputs.to(device), Y.to(device))
        loss.backward()
        optimizer.step()
        loss_epoch += loss.data
        c += 1
    loss_epoch /= c
    writer.add_scalar('training loss', loss_epoch, e+1)
#     scheduler.step()
    val_loss_epoch = validate()
    writer.add_scalar('validation loss', val_loss_epoch, e+1)
    print('Train loss {}, Val loss {}'.format(loss_epoch, val_loss_epoch))
    torch.save(myNet, os.path.join(run_path, 'latest_model'))
    if val_loss_epoch < best_val:
        torch.save(myNet, os.path.join(run_path, 'best_val_model'))

t_end = time.time()
print('time taken {}'.format(t_end-t_start))