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
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=99, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 99)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

X_train, y_train, X_test, y_test = divide_data()
print("0")
X_train = pickle.load(open(".\\data\\X_train.sav", 'rb'))
y_train = pickle.load(open(".\\data\\X_train.sav", 'rb'))
X_test = pickle.load(open(".\\data\\X_train.sav", 'rb'))
y_test = pickle.load(open(".\\data\\X_train.sav", 'rb'))

model = my_LSTM(input_size, hidden_layer_size, output_size)
#model = pickle.load(open(".\\best_model.sav", 'rb'))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)
n_epochs = 10000
lowest_loss = 10000
ll_epoch = -1
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)        
        loss = loss_fn(y_pred, y_batch)
        print(y_pred.detach().numpy()[0][0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Validation
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))
        
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

    if lowest_loss > test_rmse:
        lowest_loss = test_rmse
        pickle.dump(model, open(".\\best_model.sav", 'wb'))
        ll_epoch = epoch

print("Epoch %d Lowest loss: %.4f" % (ll_epoch, lowest_loss))
    
