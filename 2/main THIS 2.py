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
num_epochs = 20
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

sl_tab = [100, 50, 25, 10] 
hs_tab = [128, 256, 64]
nl_tab = [1, 2, 3]

best_best_loss = 9999999999999999
best_best_descr = "" 

for seq_length in sl_tab: # Length of input sequence
    for hidden_size in hs_tab: # Size of LSTM hidden state
        for num_layers in nl_tab: # Number of LSTM layers
            current_model_description = str(seq_length) + "-" + str(hidden_size) + "-" + str(num_layers) + "-"
            writeout(current_model_description + "--------------------")
            print(current_model_description)
            patience = 10
            lowest_loss = 999999999999
            ll_epoch = -1

            # Data
            dataset = LetterDataset(seq_length, '.\\data\\train_data.txt')
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            val_dataset = LetterDataset(seq_length, '.\\data\\val_data.txt')
            val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

            # Initialize model, loss function, optimizer
            model = LSTMModel(input_size, hidden_size, output_size, num_layers)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Training loop
            for epoch in range(num_epochs):
                val_loss = 0
                test_loss = 0
                model.train()
                for sequences, targets in data_loader:
                    optimizer.zero_grad()
                    outputs = model(sequences)
                    loss = criterion(outputs, targets.float())
                    loss.backward()
                    optimizer.step()

                # Validation
                model.eval()
                with torch.no_grad():
                    i = 0
                    for sequences, targets in val_data_loader:
                        outputs = model(sequences)
                        val_loss = val_loss + criterion(outputs, targets.float()).item()
                        i = i + 1
                    val_loss = val_loss / i
                    writeout(f'VAL Epoch {epoch}, Loss: {val_loss:.4f}')
                    print(f'VAL Epoch {epoch}, Loss: {val_loss:.4f}')

                if lowest_loss > val_loss:
                    lowest_loss = val_loss
                    model_name = ".\\saved_models\\" + current_model_description + "best_model.sav"
                    pickle.dump(model, open(model_name, 'wb'))
                    ll_epoch = epoch
                    patience = 10
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

def predict(model, sequence):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        sequence_tensor = torch.tensor(sequence, dtype=torch.long).unsqueeze(0)  # Add batch dimension
        output = model(sequence_tensor)
        predicted_letter = torch.argmax(output, dim=1)
    return predicted_letter.item()  # Get the predicted letter index

