import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from MyEncoderDecoder import MyEncoderDecoder
import datetime
import pickle
import numpy as np
import math
import torch.nn.functional as F
import lightning as L
from torch.optim import Adam
from contextlib import *
from torchmetrics import Accuracy

med = MyEncoderDecoder()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
global global_loss_value
global_loss_value = 9999999999

class PositionEncoding(nn.Module):
    def __init__(self, d_model=2, max_len=6):
        ## d_model = The dimension of the transformer, which is also the number of embedding values per token.
        ## max_len = maximum number of tokens we allow as input.
        ##           Since we are precomputing the position encoding values and storing them in a lookup table
        ##           we can use d_model and max_len to determine the number of rows and columns in that
        ##           lookup table.
        
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)
        embedding_index = torch.arange(start=0, end=d_model, step=2).float()
        div_term = 1/torch.tensor(10000.0)**(embedding_index / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        try:
            pe[:, 1::2] = torch.cos(position * div_term)
        except:
            pe[:, 1::2] = torch.cos(position * div_term)[:,0:-1]
        self.register_buffer('pe', pe)
        
    def forward(self, word_embeddings):
        return word_embeddings + self.pe[:word_embeddings.size(0), :]

class Attention(nn.Module): 
    def __init__(self, d_model=2):
        ## d_model = the number of embedding values per token.        
        super().__init__()
        self.d_model=d_model
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.row_dim = 0
        self.col_dim = 1

        
    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask=None):
        ## Create the query, key and values using the encodings
        ## associated with each token (token encodings)
        q = self.W_q(encodings_for_q)
        k = self.W_k(encodings_for_k)
        v = self.W_v(encodings_for_v)

        sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))
        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9)

        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)
        attention_scores = torch.matmul(attention_percents, v)
        return attention_scores

      
def move_to(obj, device):
  if torch.is_tensor(obj):
    return obj.to(device)
  elif isinstance(obj, dict):
    res = {}
    for k, v in obj.items():
      res[k] = move_to(v, device)
    return res
  elif isinstance(obj, list):
    res = []
    for v in obj:
      res.append(move_to(v, device))
    return res
  else:
    raise TypeError("Invalid type for move_to")

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

start = datetime.datetime.now()
outfile = str(start).replace(":", "").replace("-", "").replace(".", "")
outfile = ".\\output\\" + outfile + ".txt"

def writeout(line):
    f = open(outfile, "a")
    line = line + "\n"
    f.write(line)
    f.close()

learning_rate = 0.001
num_epochs = 10
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
 

class DecoderOnlyTransformer(L.LightningModule):
    
    def __init__(self, num_tokens=4, d_model=2, max_len=6):
        super().__init__()

        self.we = nn.Embedding(num_embeddings=num_tokens, 
                               embedding_dim=d_model)     
        
        self.pe = PositionEncoding(d_model=d_model, 
                                   max_len=max_len)

        self.self_attention = Attention(d_model=d_model)
        ## self.self_attention_2 = Attention(d_model=d_model)
        ## self.self_attention_3 = Attention(d_model=d_model)
        ## self.reduce_attention_dim = nn.Linear(in_features=(num_attention_heads*d_model), out_features=d_model)

        self.fc_layer = nn.Linear(in_features=d_model, out_features=num_tokens)
        self.loss = nn.CrossEntropyLoss()        
        
    def forward(self, token_ids):
                
        word_embeddings = self.we(token_ids)        
        position_encoded = self.pe(word_embeddings)
        mask = torch.tril(torch.ones((token_ids.size(dim=0), token_ids.size(dim=0)), device=self.device))

        ## Replace the 0s above the digaonal, which represent the letters
        ## we want to be masked out, with "True", and replace the ones in the lower
        ## triangle, which represent the words we want to include when we calcualte
        ## self-attention for a specific word in the output, with "False".
        mask = mask == 0
        
        self_attention_values = self.self_attention(position_encoded, 
                                                    position_encoded, 
                                                    position_encoded, 
                                                    mask=mask)

        ## self_attention_values_2 = self.self_attention_2(...)
        ## self_attention_values 3 = self.self_attention_3(...)
        ## 
        ## ...then concatenate all the self attention values...
        ##
        ## all_self_attention_values = torch.cat(self_attention_values_1, ...)
        ##
        ## ...and then run them through reduce_dim to get back to d_model values per token
        ##
        ## final_self_attention_values = self.reduce_attention_dim(all_self_attention_values)
                
        residual_connection_values = position_encoded + self_attention_values
        
        fc_layer_output = self.fc_layer(residual_connection_values)
        
        return fc_layer_output
    
    
    def configure_optimizers(self): 
        return Adam(self.parameters(), lr=0.1)
    
    
    def training_step(self, batch, batch_idx): 
        ## training_step() is called by Lightning trainer when 
        ## we want to train the model.
        input_tokens, labels = batch # collect input
        output = self.forward(input_tokens[0])
        loss = self.loss(output, labels[0])       
        return loss

    def validation_step(self, batch, batch_idx):
        input_tokens, labels = batch # collect input
        output = self.forward(input_tokens[0])
        val_loss = self.loss(output, labels[0])
        self.log('val_loss', val_loss, on_step=True, on_epoch=True)
        print("\nvalidation_loss: ", float(val_loss))
        return val_loss

d_model_table = [2, 512]
sl_tab = [50]

best_best_loss = 9999999999
best_best_descr = "" 

for seq_length in sl_tab: # Length of input sequence
    for d_model_param in d_model_table: # Size of LSTM hidden state
      with open(outfile, 'a') as f:
        with redirect_stdout(f):
                # Data
                dataset = LetterDataset(seq_length, '.\\data\\train_data.txt')
                data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                val_dataset = LetterDataset(seq_length, '.\\data\\val_data.txt')
                val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                ## First, create a model from DecoderOnlyTransformer()
                model = DecoderOnlyTransformer(num_tokens=input_size, d_model=d_model_param, max_len=seq_length)
                input_length = seq_length

                ## Use model
                #predictions = model(model_input) 
                #predicted_id = torch.tensor([torch.argmax(predictions[-1,:])]) #last character
                ## We'll store predicted_id in an array, predicted_ids, that
                ## we'll add to each time we predict a new output token.
                #predicted_ids = predicted_id

                for i in range(1):
                  # Training
                  trainer = L.Trainer(max_epochs=25, default_root_dir=".\\saved_models\\")
                  trainer.fit(model, data_loader, val_data_loader)

                  model_name = ".\\saved_models\\" + "full_best_" + str(d_model_param) + "_lightning_model.sav"
                  pickle.dump(model, open(model_name, 'wb'))
                    
                    
                  
                  

