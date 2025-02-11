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
        self.data = torch.tensor(med.encode(file.read().replace('\n', ' ')), dtype=torch.long).to(device)
        self.seq_length = seq_length

    def __len__(self):
        return int(len(self.data)) - self.seq_length

    def __getitem__(self, idx):
        sequence = self.data[int(idx):int(idx)+self.seq_length]
        encoded_target = self.data[int(idx)+1:int(idx)+self.seq_length+1]
        return sequence, encoded_target

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out)
        out = nn.functional.softmax(out, dim=1)   
        return out

import os
med = MyEncoderDecoder()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

directory = os.fsencode(".\\saved_models_second_run")
    
#for file in os.listdir(directory):
#    filename = os.fsdecode(file)
filename = "50-128-2-best_model_old.sav"
if True:
    seq_length, hidden_size, num_layers, rest = filename.split("-")
    seq_length = int(seq_length)
    hidden_size = int(hidden_size)
    num_layers = int(num_layers)
    print(filename)
    if True:
        if True:
            # Data
            test_dataset = LetterDataset(seq_length, '.\\data\\test_data.txt')
            test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # Initialize model, loss function, optimizer
            fi = ".\\saved_models_second_run\\" + filename
            model = pickle.load(open(fi, 'rb')).to(device)
            model.eval()

            accuracy_for_batches = []

            if False:
                # Validation
                with torch.no_grad():
                    accuracy = 0
                    i = 0
                    
                    for sequences, targets in test_data_loader:
                        sequences = sequences[0]
                        targets = targets[0]
                        i = i + 1
                        outputs = model(sequences)
                        max_pred = max(outputs[-1])
                        max_pred = max_pred.item()
                        for k in range(len(outputs[-1])):
                            if outputs[-1][k].item() == max_pred:
                                if k == targets[-1].item():
                                    accuracy += 1
                                    
                        

                print("Accuracy: ")
                print(accuracy/i)
            

            for a in range(1):
                tekst = " tymczasem przenoś moją duszę utęsknioną do tych pa"
                tekst = "mała mi miała mały młyn i małą miskę i małego misia"
                
                for x in range(2):
                    sequence = med.encode(tekst)
                    outputs = model(torch.tensor(sequence, dtype=torch.long).to(device))
                    
                    o = ""
                    for ch in range(len(outputs)):
                        max_pred = max(outputs[ch])
                        max_pred = max_pred.item()
                        for k in range(len(outputs[ch])):
                            if outputs[ch][k].item() == max_pred:
                                o = o + str(med.decode([k]))
                    print(o)
                    tekst = o

                    

                
                    
                
                        
                            
        

