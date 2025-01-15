import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from MyEncoderDecoder import MyEncoderDecoder
import pickle
import torch.utils.data as data
import numpy as np

med = MyEncoderDecoder()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dictionary_size = 53
hidden_layer_size = 64
output_size = 100
input_size = 100

torch.manual_seed(1)
memory = torch.rand(1,99)

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

token = "o nauka o grzeczności — podkomorzego uwagi polityczne nad modami — początek sporu o kusego i sokoła"
encoding = med.encode(token)
array = [encoding, encoding]
model = pickle.load(open(".\\best_model_2.sav", 'rb'))
model.eval()
with torch.no_grad():
    memory = torch.rand(8,99)
    y_pred = model(torch.tensor(array).to(torch.float32), memory)
    res = y_pred[0].detach().numpy()
    for el in res:
        print(med.decode(el))
        
