import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from MyEncoderDecoder import MyEncoderDecoder
import pickle
import torch.utils.data as data
import numpy as np
import math
import Variable

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
        
    X_train = torch.tensor(X_train).to(torch.int64)
    y_train = torch.tensor(y_train).to(torch.int64)
    X_test = torch.tensor(X_test).to(torch.int64)
    y_test = torch.tensor(y_test).to(torch.int64)
    
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

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

#https://stackoverflow.com/questions/77444485/using-positional-encoding-in-pytorch
#https://discuss.pytorch.org/t/transformer-example-position-encoding-function-works-only-for-even-d-model/100986/3
class my_TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, max_length, dropout=0.1):
        super(my_TransformerDecoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = self.get_positional_encoding(max_length, embedding_dim)
        
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.fc = nn.Linear(embedding_dim, vocab_size)
        
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
            
        output = self.fc(tgt)
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

#X_train, y_train, X_test, y_test = divide_data()
#print("0")
X_train = pickle.load(open(".\\data\\X_train.sav", 'rb'))
y_train = pickle.load(open(".\\data\\X_train.sav", 'rb'))
X_test = pickle.load(open(".\\data\\X_train.sav", 'rb'))
y_test = pickle.load(open(".\\data\\X_train.sav", 'rb'))

model = my_TransformerDecoder(99, 99, 1, 27, 99)
memory = torch.rand(99,99,99)
tgt_mask = []
for i in range(8):
    tgt_mask.append(generate_square_subsequent_mask(sz=99))
#memory = torch.rand(7,99)
#model = pickle.load(open(".\\best_model.sav", 'rb'))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=99)
n_epochs = 10000
lowest_loss = 10000
ll_epoch = -1
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch, memory, tgt_mask)
        loss = loss_fn(y_pred[0], y_batch)
        #loss = Variable(loss, requires_grad = True)
        loss = torch.tensor(loss, requires_grad=True).to(torch.float32)
        optimizer.zero_grad()
        print(loss)
        loss.backward()
        optimizer.step()
        
    # Validation
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train, memory, tgt_mask)
        train_rmse = np.sqrt(loss_fn(y_pred[0], y_train))
        y_pred = model(X_test, memory, tgt_mask)
        test_rmse = np.sqrt(loss_fn(y_pred[0], y_test))
        
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

    if lowest_loss > test_rmse:
        lowest_loss = test_rmse
        pickle.dump(model, open(".\\best_model_3.sav", 'wb'))
        ll_epoch = epoch

print("Epoch %d Lowest loss: %.4f" % (ll_epoch, lowest_loss))
    
