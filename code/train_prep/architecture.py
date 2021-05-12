import torch
import torch.nn as nn
from einops import rearrange
# BATCH FIRST TimeDistributed layer

class HybridModel(nn.Module):
    def __init__(self,num_emotions, num_chunks=8):
        super().__init__()
        
        
        #Set of convolution filters to extract the relevant informations from each chunk
        self.convBlock = nn.Sequential(
            # 1. conv block
            nn.Conv2d(in_channels=1,
                                   out_channels=16,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1
                                  ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.4),
        
        
            # 2. conv block
            nn.Conv2d(in_channels=16,
                                   out_channels=32,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1
                                  ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.4),

            # 3. conv block
            nn.Conv2d(in_channels=32,
                                   out_channels=64,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1
                                  ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.4),


            # 4. conv block
            nn.Conv2d(in_channels=64,
                                   out_channels=128,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1
                                  ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.4)
        )
        
        # LSTM block for aggregating the information obtained from the convolution steps
        hidden_size = 64
        self.lstm = nn.LSTM(input_size=128,hidden_size=hidden_size,bidirectional=True, batch_first=True) 
        self.dropout_lstm = nn.Dropout(p=0.3)
        
        # Linear softmax layer
        self.out_linear = nn.Linear(2*hidden_size,num_emotions)
        
        #Attention linear layer to extract the relevant passages from the LSTM
        self.lstm_linear_attention = nn.Linear(num_chunks, 1)
                        

    def forward(self,x):
        
        batch_size = x.shape[0]
        
        #For the convolution filter pass, we are only interested in regressing the window of 128*128. 
        #Hence, we group together the batch and the chunks.
        x = rearrange(x, "b t c h w -> (b t) c h w")
        
        # reduces the features 
        conv_embedding = rearrange(self.convBlock(x), "(b t) c h w -> b t (c h w)", b = batch_size)

        # Process the lstm over the chunks
        lstm_embedding, _ = self.lstm(conv_embedding)
        lstm_embedding = self.dropout_lstm(lstm_embedding)
        
        #Perform attention on the resulting output
        lstm_output = self.lstm_linear_attention(rearrange(lstm_embedding, ' b p d -> b d p'))[:,:,0]
        
        output_logits = self.out_linear(lstm_output)
                
        output_softmax = nn.functional.softmax(output_logits,dim=1)
        
        return output_logits, output_softmax


class PytorchScaler():
    """
    Scale each channel for values between [0, 1].
    """
    def Scaler(self, tensor):
        scale = 1.0 / (tensor.max(dim=1, keepdim=True)[0] - tensor.min(dim=1, keepdim=True)[0]+1) 
        tensor.mul_(scale).sub_(tensor.min(dim=1, keepdim=True)[0])
        return tensor
        
    def __call__(self, tensor):
        
        b,t,c,h,w = tensor.shape
        
        #We normalize each batch
        tensor = rearrange(tensor, 'b t c h w -> b (t c h w)')
        tensor = self.Scaler(tensor)
        tensor = rearrange(tensor, 'b (t c h w) -> b t c h w', t = t, c = c, h = h, w = w)
        
        return tensor

def Dataloader(X_train, X_val, Y_train, Y_val, device = 'cpu'):
    #? Load the data into torch tensors

    print("Dataloader")
    X_train_tensor = torch.tensor(X_train,device=device).float()
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long,device=device)
    
    X_val_tensor = torch.tensor(X_val,device=device).float()
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.long,device=device)
    
    #? Scale the data
    scaler = PytorchScaler()
    X_train_tensor = scaler(X_train_tensor)
    X_val_tensor = scaler(X_val_tensor)
 
    return X_train_tensor, Y_train_tensor, X_val_tensor, Y_val_tensor



def make_train_step(model, loss_fnc, optimizer):
    def train_step(X,Y):
        # set model to train mode
        model.train()
        # forward pass
        output_logits, output_softmax = model(X)
        predictions = torch.argmax(output_softmax,dim=1)
        accuracy = torch.sum(Y==predictions)/float(len(Y))
        # compute loss
        loss = loss_fnc(output_logits, Y)
        # compute gradients
        loss.backward()
        # update parameters and zero gradients
        optimizer.step()
        optimizer.zero_grad()
        return loss.item(), accuracy*100
    return train_step


def make_validate_fnc(model,loss_fnc):
    def validate(X,Y):
        with torch.no_grad():
            model.eval()
            output_logits, output_softmax = model(X)
            predictions = torch.argmax(output_softmax,dim=1)
            accuracy = torch.sum(Y==predictions)/float(len(Y))
            loss = loss_fnc(output_logits,Y)
        return loss.item(), accuracy*100, predictions
    return validate