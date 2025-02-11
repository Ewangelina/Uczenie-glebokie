import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from MyEncoderDecoder import MyEncoderDecoder
import datetime
import pickle

med = MyEncoderDecoder()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

start = datetime.datetime.now()
outfile = str(start).replace(":", "").replace("-", "").replace(".", "")
outfile = ".\\output\\" + outfile + ".txt"

def writeout(line):
    f = open(outfile, "a")
    line = line + "\n"
    f.write(line)
    f.close()

learning_rate = 0.001
num_epochs = 8000
batch_size = 1
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

sl_tab = [10, 100, 50, 25] 
hs_tab = [64, 128, 256]
nl_tab = [1, 2, 3]

sl_tab = [100, 50] 
hs_tab = [128, 256]
nl_tab = [2]

best_best_loss = 9999999999999999
best_best_descr = ""
patience_difference = 0.1
skip_models = 1

for hidden_size in hs_tab: # Size of LSTM hidden state
    for seq_length in sl_tab: # Length of input sequence
        for num_layers in nl_tab: # Number of LSTM layers
            if skip_models > 0:
                skip_models -= 1
                model = pickle.load(open(".\\saved_models\\100-128-2-best_model.sav", 'rb'))
                model.to(device)
            else:
                model = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)

            current_model_description = str(seq_length) + "-" + str(hidden_size) + "-" + str(num_layers) + "-"
            writeout(current_model_description + "--------------------")
            print(current_model_description)
            patience = 3
            lowest_loss = 999999999999
            ll_epoch = -1

            # Initialize model, loss function, optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Data
            dataset = LetterDataset(seq_length, '.\\data\\train_data.txt')
            print(len(dataset))
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            val_dataset = LetterDataset(seq_length, '.\\data\\val_data.txt')
            print(len(val_dataset))
            val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

            # Training loop
            for epoch in range(num_epochs):
                model.train()
                for sequences, targets in data_loader:
                    sequences = sequences[0]
                    targets = targets[0]
                    optimizer.zero_grad()
                    outputs = model(sequences)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for sequences, targets in val_data_loader:
                        sequences = sequences[0]
                        targets = targets[0]
                        outputs = model(sequences)
                        val_loss = val_loss + criterion(outputs, targets)
                    writeout(f'VAL Epoch {epoch}, Loss: {val_loss:.4f}')
                    print(f'VAL Epoch {epoch}, Loss: {val_loss:.4f}')

                if lowest_loss > val_loss + patience_difference:
                    lowest_loss = val_loss
                    model_name = ".\\saved_models\\" + current_model_description + "best_model.sav"
                    pickle.dump(model, open(model_name, 'wb'))
                    ll_epoch = epoch
                    patience = 3
                else:
                    patience = patience - 1
                    if patience <= 0:
                        line = "Lowest loss at epoch " + str(ll_epoch)
                        writeout(line)
                        print(line)
                        break

print("Training done!")
line = "Best model: " + best_best_descr + " Loss: " + str(best_best_loss)
print(line)
writeout(line)

