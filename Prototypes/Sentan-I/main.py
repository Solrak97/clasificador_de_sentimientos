import pandas as pd
from processing import *
from Dias_Model import Dias_Model

import torch
#from Sentan_Model import Sentan_Model

data = pd.read_pickle('data.pkl')
data = data.drop(['File name', 'Duration'], axis=1)

raw_y = data['Emotion'] 
raw_x = data.drop(['Emotion'], axis=1)



x = x_flat(raw_x)
x = x_to_tensor(x)
y = encode(raw_y)






# Base Dias test 1

from torch.optim import Adam
from torch import nn

x_train = x
y_train = torch.from_numpy(y).type(torch.float32)



# Data transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Dias_Model()


# Hiperparametros
INIT_LR = 1e-3
EPOCHS = 700
VAL_SIZE = 10000
TRAIN_SIZE = 1440


# Optimizer y funcion de perdida
opt = Adam(model.parameters(), lr=INIT_LR)
lossFn = nn.CrossEntropyLoss()

# Historial de entrenamiento
loss_hist = []
train_acc_hist = []
val_acc_hist = []
loss = 0

# Entrenamiento del modelo
model.train()
for epoch in range(0, EPOCHS):
    # Training

    x_train.to(device)
    y_train.to(device)

    opt.zero_grad()

    pred = model(x_train)


    _loss = lossFn(pred, y_train)
    loss_dif = loss - _loss
    loss = _loss
    loss.backward()
    opt.step()

    
    train_correct = (torch.argmax(pred, dim=1) == torch.argmax(
        y_train, 1)).type(torch.float).sum().item()

    
    train_acc = train_correct / TRAIN_SIZE

    # Report

    print(f'''
    
    Epoch #{epoch}
    Loss                {loss}
    Loss Dif:           {loss_dif}
    Train Correct:      {train_correct}
    Train Acc:          {train_acc}

    ''')
