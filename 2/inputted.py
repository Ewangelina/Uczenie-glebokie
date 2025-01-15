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

class my_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=99, hidden_size=99, num_layers=1, batch_first=True)
        self.linear = nn.Linear(99, 99)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

token = "mała małgosia miała mały młyn i małego kotka i małego braciszka i mały samochodzik i małą piłeczkę "
model = pickle.load(open(".\\best_model_hs_99.sav", 'rb'))

array = [med.encode(token), med.encode(token)]
y_pred = model(torch.tensor(array).to(torch.float32))
res = y_pred.detach().numpy()
print(res)
for el in res:
    print(med.decode(el))
    
