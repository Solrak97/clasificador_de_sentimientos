from torch import nn
from torch.optim import Adam
import pandas as pd


from pre_processing import to_tensor, to_labels, load_data, split
from model_training import train
from model_evaluation import confussion_matrix


from Dias_Model import Dias_Model
from Sentan_Model import Sentan_Model



data_path = 'data.pkl'

features, labels, classes = load_data(data_path)
#k_folds = split(features, labels)

lbls = to_labels(labels, classes)
confussion_matrix(lbls, lbls, classes)


'''
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
'''