from torch import nn
from torch.optim import Adam
import pandas as pd
from data_processing import *
from Dias_Model import Dias_Model

from model_training import train
#from Sentan_Model import Sentan_Model

data = pd.read_pickle('data.pkl')
data = data.drop(['File name', 'Duration'], axis=1)

raw_y = data['Emotion']
raw_x = data.drop(['Emotion'], axis=1)


x_train, x_test, y_train, y_test = split_and_transform(raw_x, raw_y)


EPOCHS = 700
INIT_LR = 1e-3
TRAIN = (x_train, y_train)
TEST = (x_test, y_test)
MODEL = Dias_Model()
OPTIMIZER = Adam(MODEL.parameters(), lr=INIT_LR)
LOSS_FN = nn.CrossEntropyLoss()
VERBOSE = True

train(MODEL, EPOCHS, TRAIN, TEST, OPTIMIZER, LOSS_FN, VERBOSE)

torch.save(MODEL.state_dict(), 'Prototypes/Sentan-I/base_model.pt')