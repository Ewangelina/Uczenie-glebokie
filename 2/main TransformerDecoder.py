import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from MyEncoderDecoder import MyEncoderDecoder
import pickle
import torch.utils.data as data
import numpy as np

med = MyEncoderDecoder()

def divide_data():
    file = open('.\\data\\pairs.csv', 'r', encoding="utf-8")
    i = 1
    percent_of_test = int(445541 / 89000)

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    lines = file.readlines()
    for line in lines:
        x = med.encode(line[0:99])
        y = med.encode(line[100:199])
        if i == percent_of_test:
            X_test.append(x)
            y_test.append(y)
            i = 1
        else:
            X_train.append(x)
            y_train.append(y)
        i = i + 1

    X_train = torch.tensor(X_train).to(torch.float32)
    y_train = torch.tensor(y_train).to(torch.float32)
    X_test = torch.tensor(X_test).to(torch.float32)
    y_test = torch.tensor(y_test).to(torch.float32)
    
    pickle.dump(X_train, open(".\\data\\X_train.sav", 'wb'))
    pickle.dump(y_train, open(".\\data\\y_train.sav", 'wb'))
    pickle.dump(X_test, open(".\\data\\X_test.sav", 'wb'))
    pickle.dump(y_test, open(".\\data\\y_test.sav", 'wb'))

    print(len(X_train))
    print(len(X_test))

    return X_train, y_train, X_test, y_test

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dictionary_size = 53
hidden_layer_size = 64
output_size = 100
input_size = 100

torch.manual_seed(1)

class my_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=99, nhead=9)
        self.lstm = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.linear = nn.Linear(99, 99)
    def forward(self, x, memory):
        x = self.lstm(x, memory)
        x = self.linear(x)
        return x, memory

#X_train, y_train, X_test, y_test = divide_data()
#print("0")
X_train = pickle.load(open(".\\data\\X_train.sav", 'rb'))
y_train = pickle.load(open(".\\data\\X_train.sav", 'rb'))
X_test = pickle.load(open(".\\data\\X_train.sav", 'rb'))
y_test = pickle.load(open(".\\data\\X_train.sav", 'rb'))

model = my_LSTM(input_size, hidden_layer_size, output_size)
memory = torch.rand(8,99)
#memory = torch.rand(7,99)
#model = pickle.load(open(".\\best_model_2.sav", 'rb'))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)
n_epochs = 10000
lowest_loss = 10000
ll_epoch = -1
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        memory = torch.rand(8,99)
        y_pred = model(X_batch, memory)
        loss = loss_fn(y_pred[0], y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Validation
    model.eval()
    with torch.no_grad():
        memory = torch.rand(8,99)
        y_pred = model(X_train, memory)
        train_rmse = np.sqrt(loss_fn(y_pred[0], y_train))
        memory = torch.rand(8,99)
        y_pred = model(X_test, memory)
        test_rmse = np.sqrt(loss_fn(y_pred[0], y_test))
        
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

    if lowest_loss > test_rmse:
        lowest_loss = test_rmse
        pickle.dump(model, open(".\\best_model_2.sav", 'wb'))
        ll_epoch = epoch

print("Epoch %d Lowest loss: %.4f" % (ll_epoch, lowest_loss))
    
