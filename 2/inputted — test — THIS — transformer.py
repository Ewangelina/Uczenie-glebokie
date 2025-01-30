import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from MyEncoderDecoder import MyEncoderDecoder
import datetime
import pickle
import numpy as np
import math
import torch.nn.functional as F

med = MyEncoderDecoder()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

start = datetime.datetime.now()
outfile = str(start).replace(":", "").replace("-", "").replace(".", "")
outfile = ".\\output\\" + outfile + ".txt"

def writeout(line):
    f = open(outfile, "a")
    line = line + "\n"
    f.write(line)
    f.close()

learning_rate = 0.001
num_epochs = 20
batch_size = 64
input_size = 54  # Number of unique letters
output_size = 1  # Number of target letters

class LetterDataset(Dataset):
    def __init__(self, seq_length, source):
        super(LetterDataset, self).__init__()
        file = open(source, 'r', encoding="utf-8")
        self.data = file.read().replace('\n', ' ')
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        sequence = med.encode(self.data[idx:idx+self.seq_length])
        target = med.encode(self.data[idx+self.seq_length])
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(target, dtype=torch.long)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads, num_layers=1):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.positional_encoding = self.get_positional_encoding(input_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, dropout=0.01)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, tgt, memory, tgt_mask, tgt_is_causal=True):
        ae = self.embedding(tgt)
        s = tgt.size(-1)
        for el in ae:
            for elel in el:
                for elelel in elel:
                    elelel = elelel * math.sqrt(s)
        tgt = ae + self.positional_encoding
        tgt = F.dropout(tgt, p=0.1)
        
        for layer in self.layers:
            tgt = layer(tgt, memory)
            
        output = self.fc(tgt[:, -1, :])
        return output

    def get_positional_encoding(self, max_length, embedding_dim):
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(np.log(10000.0) / embedding_dim))
        positional_encoding = torch.zeros(max_length, embedding_dim)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        try:
            positional_encoding[:, 1::2] = torch.cos(position * div_term)
        except:
            positional_encoding[:, 1::2] = torch.cos(position * div_term)[:,0:-1]
        return positional_encoding

    

sl_tab = [54] 
hs_tab = [128]
nl_tab = [1]
hd_tab = [1]

best_best_loss = 9999999999999999
best_best_descr = "" 

for seq_length in sl_tab: # Length of input sequence
    for hidden_size in hs_tab: # Size of LSTM hidden state
        memory = torch.rand(batch_size, seq_length, hidden_size)
        tgt_mask = []
        for i in range(batch_size):
            tgt_mask.append(generate_square_subsequent_mask(sz=hidden_size))
        for num_layers in nl_tab: # Number of LSTM layers
            for heads in hd_tab:
                val_dataset = LetterDataset(seq_length, '.\\data\\test_data.txt')
                val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

                # Initialize model, loss function, optimizer
                model = pickle.load(open(".\\saved_models\\54-128-1-1-best_model_2.sav", 'rb'))
                model.eval()
                accuracy_for_batches = []

                

                with torch.no_grad():
                    for sequences, targets in val_data_loader:
                        accuracy = 0
                        i = 0
                        outputs = model(sequences, memory, tgt_mask)
                        for j in range(len(outputs)):
                            i = i + 1
                            if round(float(outputs[j])) == int(targets[j]):
                                accuracy = accuracy + 1
                        accuracy_for_batches.append(accuracy/i)

            print("Minimum: ")
            print(min(accuracy_for_batches))
            print("Max: ")
            print(max(accuracy_for_batches))
            print("AVG: ")
            print(sum(accuracy_for_batches) / float(len(accuracy_for_batches)))
