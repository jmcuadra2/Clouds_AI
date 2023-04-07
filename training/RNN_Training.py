import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import glob
from torch.utils.data import Dataset, DataLoader
import time
import torchvision.transforms as transforms


# ## Loading and preparing the data


# dataset class, is in charge of grabbing the numpy files where the simulation data has been saved, applying the 
# sliding window and load them. __init__, __len__ and __getitem__ are functions that must be named like that 
#by pytorch command, as they will be used by the dataloader function.
class Dataset(Dataset):

    def __init__(self, path,  n_past = 10, n_future = 1, spheres = 35, transform=None):
        self.transform = transform
        self.n_past = n_past
        self.n_future = n_future
        numpy_vars = []
        for np_name in glob.glob(path + '/*.npy'):
            data = np.load(np_name)
            if data.shape[1] != spheres:
                data = np.hstack([data, np.zeros([data.shape[0], spheres - data.shape[1], 3])])
            numpy_vars.append(  data  )

        data = np.concatenate(numpy_vars , axis=0)
        data_len = data.shape[0]
        data = data.reshape((data_len, -1))
        train_x, train_y = [], []
        for i in range(n_past, data_len):
            train_x.append(data[i - n_past: i])
            train_y.append(data[i : i + n_future])
        self.X = torch.tensor(train_x) 
        self.Y = torch.tensor(train_y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        train_x = self.X[idx]
        train_y = self.Y[idx]
        sample = {'input': self.X[idx], 'target':  self.Y[idx]}
        sample = (self.X[idx].float(), self.Y[idx].float())

        if self.transform:
            
            sample = self.transform(sample)

        return sample
    

train_data = Dataset(path = '/path')


#n_past: window size
#n_future: number of subsequent stages to be predicted
def load_dataset(data, n_past = 10, n_future = 1):
    train_x, train_y = [], []
    for i in range(n_past, data.shape[1]):
        train_x.append(data[: , i - n_past: i])
        train_y.append(data[:, i : i + n_future])
    return torch.tensor(train_x) , torch.tensor(train_y)

data_1 = np.load("path/cumulus_00.npy")
plt.plot(data_1[0:1000,10,0])


dataset_loader = torch.utils.data.DataLoader(train_data,
                                             batch_size=24, shuffle=True)



#dataset size (no. batches per batch size)
len(dataset_loader)*64


#We paint a batch to see its size, batch first, then sequence and then velocity data (batch, seq, X)
for X, Y in dataset_loader:
        X = X
        Y = Y.squeeze()
        print(X.shape)
        break





#validation dataset
test_data = Dataset(path = 'path/test/')

test_loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=64, shuffle=False)





#size of evaluation dataset
len(test_loader)*64





# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


### Architectures

#RNN simple
class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)

        #Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)
        

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        #out = out.contiguous().view(-1, self.hidden_dim)
        out = out[:,-1]
        out = self.fc(out)
        
        return out
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
         # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden





#LSTM
class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(LSTM, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim,
                            num_layers=n_layers, batch_first=True)
           
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
       # self.drop = nn.Dropout(p=0.2)
        #self.fc2 = nn.Linear(210, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)

        #Initializing hidden state for first input using method defined below
        hidden, cell = self.init(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, (hidden, cell) = self.lstm(x, (hidden, cell))
       # print('out: ', out.shape)
        out = out[:,-1]
        #print('out: ', out.shape)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        #out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        #out = self.drop(out)
        #out = self.fc2(out)
        
        return out
    
    def init(self, batch_size):

        hidden = torch.zeros( self.n_layers, batch_size, self.hidden_dim).to(device)
        cell = torch.zeros( self.n_layers, batch_size, self.hidden_dim).to(device)

        return hidden, cell
    





#GRU
class GRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(GRU, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_dim,
                            num_layers=n_layers, batch_first=True)
           
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)

        #Initializing hidden state for first input using method defined below
        hidden = self.init(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.gru(x, hidden)
       # print('out: ', out.shape)
        out = out[:,-1]
        #print('out: ', out.shape)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        #out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        out = self.fc2(out)
        
        return out
    
    
    def init(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
         # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden
    





#FF
class FF(nn.Module):
    def __init__(self):
        super(FF, self).__init__()

        #Defining the layers
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(105*7, 210*7)
        self.fc2 = nn.Linear(210*7,420*7)
        self.fc3 = nn.Linear(420*7,210*7)
        self.fc4 = nn.Linear(210*7,105)
        
        #Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    
    def forward(self, x):
        x = self.flat(x)
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        x = self.fc4(x)

        return x





#customised error function, 
#has been tested but not implemented for final training as the results did not improve the MSE error.
def my_loss(output, target): 
    loss_all = torch.mean((output - target)**2)
    peso = 0.02 * abs(torch.mean(output))
    loss = peso + loss_all
    return loss





modelo = 'LSTM' #selected model
if modelo == 'LSTM':
    model = LSTM(input_size=105, output_size=105, hidden_dim=350, n_layers=5)
    
if modelo == 'RNN':
    model = RNN(input_size=105, output_size=105, hidden_dim=350, n_layers=5)
    
if modelo == 'GRU':
    model = GRU(input_size=105, output_size=105, hidden_dim=350, n_layers=5)
    
if modelo == 'FF':
    model = FF()
    
model = model.to(device)

n_epochs = 30 
lr=0.001

criterion = nn.MSELoss()
#criterion = my_loss()
#optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,750,1500,1750,2000,2225,2500,2900,3200], gamma=0.85)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# ## Training and eval


from ignite.handlers import FastaiLRFinder
from ignite.engine import create_supervised_trainer, create_supervised_evaluator

trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

lr_finder = FastaiLRFinder()
to_save = {"model": model, "optimizer": optimizer}

with lr_finder.attach(trainer, to_save=to_save, num_iter=None, start_lr=1e-6, end_lr=1) as trainer_with_lr_finder:
    trainer_with_lr_finder.run(dataset_loader)

# Get lr_finder results
lr_finder.get_results()

# Plot lr_finder results (requires matplotlib)
lr_finder.plot(skip_start=0, skip_end=0)

# get lr_finder suggestion for lr
lr = lr_finder.lr_suggestion()
print('Best lr: ', lr)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')






best = 2.
best_train = 2.
PATH = "path/lstm_35_5_350.pth"#save best eval result
PATH_train = "path/lstm_35_5_350_train.pth" #save best train result

def train(model=model, n_epochs=n_epochs, best=best, best_train = best_train):
    history_train = []
    history_test = []
    for epoch in range(1, n_epochs + 1):
        print('LR = ', get_lr(optimizer))

        running_loss = 0
        running_test_loss = 0

        time_0 = time.time()
        for X, Y in dataset_loader:
            X = X.to(device)
            Y = Y.squeeze().to(device)
            optimizer.zero_grad()
            output = model( X.float() )
            output = output.to(device)
            loss = criterion(output.float(), Y.float())
            #loss = my_loss(output.float(), Y.float())
            running_loss += loss
            loss.backward() # Does backpropagation and calculates gradients
            optimizer.step() # Updates the weights accordingly

        time_1 = time.time()
        elapsed = time_1 - time_0
        print('time =', elapsed/len(dataset_loader), 's/batch')
        print('total time =', elapsed, 's')
        if epoch%2 == 0:
            print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
            print("Loss: {:.4f}".format(running_loss))
        history_train.append(running_loss/5.)
        with torch.no_grad():
            for X_test, Y_test in test_loader:
                X_test = X_test.to(device)
                Y_test = Y_test.squeeze().to(device)
                output = model( X_test.float() )
                output = output.to(device)
                test_loss = criterion(output.float(), Y_test.float())
                running_test_loss += test_loss
            print("Test loss: {:.4f}".format(running_test_loss))
            history_test.append(running_test_loss)
        scheduler.step()
        if running_loss < best:
            best = running_test_loss
            torch.save(model.state_dict(), PATH)
            print('Guardando')
        if running_test_loss < best_train:
            best_train = running_loss
            torch.save(model.state_dict(), PATH_train)
            print('Guardando')
    print("Test loss: {:.4f}".format(running_test_loss))
    fig = plt.figure()
    plt.plot(history_train, label = 'Train')
    plt.plot(history_test, label = 'Test')
    fig.suptitle('LSTM training (loss vs epoch)', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend()
    fig.savefig("vector_graph_4.svg", format = 'svg', dpi=600)
    plt.show()





train(model = model,  n_epochs=50)


# ## Train iteratively for different number of hidden layers




for i in range(4,5):
    model = LSTM(input_size=105, output_size=105, hidden_dim=400, n_layers=i)
    model = model.to(device)
    
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    lr_finder = FastaiLRFinder()
    to_save = {"model": model, "optimizer": optimizer}

    with lr_finder.attach(trainer, to_save=to_save, num_iter=None, start_lr=1e-5, end_lr=10.0) as trainer_with_lr_finder:
        trainer_with_lr_finder.run(dataset_loader)

    lr_finder.get_results()
    lr_finder.plot(skip_start=0, skip_end=3)
    lr = lr_finder.lr_suggestion()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print('Nº capas: ', i)
    print('Lr: ', lr)
    train(model = model,  n_epochs=3)





plt.plot(history_train, label = 'Train')
plt.plot(history_test, label = 'Test')
plt.legend()


plt.plot(np.array(history_train)**-1, label = 'Train')
plt.plot(np.array(history_test)**-1, label = 'Test')
plt.legend()
