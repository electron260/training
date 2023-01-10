import torch
import torchaudio.transforms as T
import torchaudio.functional as F
import time

#from dataset import WakeWordDataset
from dataset.LSTMmodel import LSTMWakeWord
from typing import Tuple
from dataset.MFCCdataset import MFCC
import torchaudio



def get_featurizer(sample_rate):
    return MFCC(sample_rate=sample_rate)



@torch.no_grad()
def predict(model, input:torch.Tensor, class_mapping:Tuple[int])->int:
    model.eval()
    prob = model(input)
    prediction = int(prob>0.5)
    return prediction

class LSTMInference:
    def __init__(self,class_mapping:Tuple[int]=[0, 1]) -> None:
        self.model_cnn = LSTMWakeWord()
        self.state_dict = torch.load("state_dict_model.pt",map_location=torch.device('cpu'))
        self.model_cnn.load_state_dict(self.state_dict)
        self.class_mapping=class_mapping

        self.mfcc_transform = get_featurizer(16000)
       
    

    def get_prediction(self,x:torch.Tensor)->int:

        
        
        print(x.size())
        x= F.resample(x, 44100, 16000)
        print(x.size())
        mfcc = self.mfcc_transform(x)
        print(mfcc.size())

        mfcc = mfcc.transpose(0,1).transpose(0,2)
        print(mfcc.size())
        # if x.shape[1] > 16000:
        #     x = x[:, :16000]
        # elif x.shape[1] < 16000:
        #     num_missing_samples = 16000-x.shape[1]
        #     last_dim_padding = (0, num_missing_samples)
        #     x = torch.nn.functional.pad(x, last_dim_padding)
       
    
        prediction =predict(self.model_cnn,mfcc,self.class_mapping)
        return prediction


