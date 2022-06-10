import pandas as pd
from processing import *
#from Dias_Model import Dias_Model
#from Sentan_Model import Sentan_Model

data = pd.read_pickle('data.pkl')
data = data.drop(['File name', 'Duration'], axis=1)

raw_y = data['Emotion'] 
raw_x = data.drop(['Emotion'], axis=1)



x = x_flat(raw_x)
y = encode(raw_y)