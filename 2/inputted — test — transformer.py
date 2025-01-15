import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from MyEncoderDecoder import MyEncoderDecoder
import pickle
import torch.utils.data as data
import numpy as np

def my_split(text):
    inp = text[:99]
    outp = text[100:199]
    return inp, outp

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
    

med = MyEncoderDecoder()
file = open('.\\data\\pairs.csv', 'r', encoding="utf-8")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = pickle.load(open(".\\best_model_2.sav", 'rb'))

torch.manual_seed(1)
memory = torch.rand(1,99)


    
sw = True
token = []
out_arr = []
acc_array = []
acc_end = []
acc_end_int = []
for line in file:
    inp, out = my_split(line)
    if sw:
        token = []
        out_arr = []
        token.append(med.encode(inp))
        out_arr.append(med.encode(out))
        sw = False
    else:
        sw = True
        token.append(med.encode(inp))
        out_arr.append(med.encode(out))
        memory = torch.rand(8,99)
        y_pred = model(torch.tensor(token).to(torch.float32), memory)
        res = y_pred[0].detach().numpy()
        for j in range(2):
            accuracy = 0
            for i in range(99):
                if round(res[j][i]) == out_arr[j][i]:
                    accuracy = accuracy + 1
            if round(res[j][98]) == out_arr[j][98]:
                acc_end.append(1)
            else:
                acc_end.append(0)

            if int(res[j][98]) == out_arr[j][98]:
                acc_end_int.append(1)
            else:
                acc_end_int.append(0)
            acc_array.append(float(accuracy/99))

print("Minimum: ")
print(min(acc_array))
print("Max: ")
print(max(acc_array))
print("AVG: ")
print(sum(acc_array) / float(len(acc_array)))
print("END--------------------")
print("AVG: ")
print(sum(acc_end) / float(len(acc_end)))
print("END--int------------------")
print("AVG: ")
print(sum(acc_end_int) / float(len(acc_end_int)))
                    
                    
file.close()       
    
