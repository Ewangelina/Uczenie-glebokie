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
output_size = 54  # Number of target letters

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
        correct = torch.zeros(output_size)
        correct[med.encode(self.data[idx+self.seq_length])] = 1
        return torch.tensor(sequence, dtype=torch.long), correct


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
        probabilities = nn.functional.softmax(out, dim=1)
        return probabilities

sl_tab = [100] 
hs_tab = [128]
nl_tab = [1]


med = MyEncoderDecoder()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for seq_length in sl_tab: # Length of input sequence
    for hidden_size in hs_tab: # Size of LSTM hidden state
        for num_layers in nl_tab: # Number of LSTM layers
            # Data
            test_dataset = LetterDataset(seq_length, '.\\data\\test_data.txt')
            test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

            # Initialize model, loss function, optimizer
            model = pickle.load(open(".\\saved_models\\100-128-1-best_model.sav", 'rb'))
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
                            max_ret = max(outputs[j])
                            max_target = max(targets[j])
                            for k in range(len(outputs[j])):
                                if max_ret == outputs[j][k]:
                                    if max_target == targets[j][k]:
                                        accuracy = accuracy + 1
                        print(accuracy)
                        accuracy_for_batches.append(accuracy/i)


                print("Minimum: ")
                print(min(accuracy_for_batches))
                print("Max: ")
                print(max(accuracy_for_batches))
                print("AVG: ")
                print(sum(accuracy_for_batches) / float(len(accuracy_for_batches)))

            
            #"stały nad niemnem; napole"
            #n ,paząłwipz  pzebapęwgob

            #"tały nad niemnem; napoleo"
            #—ptc, żrcjetu zz ęrezopes
            for a in range(2):
                tekst = " wojna niechybna! kiedy z poselstwem tajemnem tu biegłem, wojsk forpoczty już stały nad niemnem; nap"
                
                for x in range(25):
                    sequence = med.encode(tekst)
                    outputs = model(torch.tensor([sequence, sequence], dtype=torch.long))
                    
                    o = ""
                    
                    ix = 0
                    max_val = outputs[a][0]
                    for x in range(len(outputs)):
                        if outputs[a][x] > max_val:
                            ix = x
                            max_val = outputs[a][x]
                        o = str(med.decode([ix]))

                    tekst = tekst[1:] + o

                print(tekst)
                    
                
                        
                            
        

