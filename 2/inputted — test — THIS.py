import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from MyEncoderDecoder import MyEncoderDecoder
import datetime
import pickle

learning_rate = 0.001
num_epochs = 8000
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
        return torch.tensor(sequence, dtype=torch.long).to(device), torch.tensor(target, dtype=torch.long).to(device)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Get the last time step output
        return out

sl_tab = [25] 
hs_tab = [256]
nl_tab = [2]


med = MyEncoderDecoder()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for seq_length in sl_tab: # Length of input sequence
    for hidden_size in hs_tab: # Size of LSTM hidden state
        for num_layers in nl_tab: # Number of LSTM layers
            # Data
            test_dataset = LetterDataset(seq_length, '.\\data\\test_data.txt')
            test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

            # Initialize model, loss function, optimizer
            model = pickle.load(open(".\\saved_models\\25-256-2-val_best_model — kopia.sav", 'rb'))
            model.eval()

            accuracy_for_batches = []

            if False:
                # Validation
                with torch.no_grad():
                    
                    for sequences, targets in test_data_loader:
                        accuracy = 0
                        i = 0
                        outputs = model(sequences)
                        for j in range(len(outputs)):
                            i = i + 1
                            if round(float(outputs[j])) == int(targets[j]):
                                accuracy = accuracy + 1
                        accuracy_for_batches.append(accuracy/i)

            tekst = "mała mi miała mały młyn i"
            #"stały nad niemnem; napole"
            #n ,paząłwipz  pzebapęwgob

            #"tały nad niemnem; napoleo"
            #—ptc, żrcjetu zz ęrezopes
            for x in range(25):
                sequence = torch.tensor([med.encode(tekst)], dtype=torch.long).to(device)
                outputs = model(sequence)

                o = ""
                for el in outputs:
                    o = o + str(med.decode(el))
                tekst = tekst[1:] + o
            print(tekst)
            print("Minimum: ")
            print(min(accuracy_for_batches))
            print("Max: ")
            print(max(accuracy_for_batches))
            print("AVG: ")
            print(sum(accuracy_for_batches) / float(len(accuracy_for_batches)))
                    
                        
    
