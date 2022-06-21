from torch import nn
from torch.optim import Adam
import pandas as pd
from data_processing import *
from Dias_Model import Dias_Model

from model_training import train
from Sentan_Model import Sentan_Model

data_path = 'data.pkl'
features, labels, classes = load_data(data_path)
x_train, x_val, y_train, y_val = split_and_tensor(features, labels)


EPOCHS = 700
INIT_LR = 1e-3
TRAIN = (x_train, y_train)
TEST = (x_val, y_val)
MODEL = Sentan_Model()
OPTIMIZER = Adam(MODEL.parameters(), lr=INIT_LR)
LOSS_FN = nn.CrossEntropyLoss()
VERBOSE = True

train(MODEL, EPOCHS, TRAIN, TEST, OPTIMIZER, LOSS_FN, VERBOSE)

torch.save(MODEL.state_dict(), 'Prototypes/Sentan-I/SaveModel/base_model.pt')