import torch
import torchaudio.transforms as T
import torchaudio.functional as F
import time

#from dataset import WakeWordDataset
from dataset.model import CNNetwork
from typing import Tuple

import torchaudio

@torch.no_grad()
def predict(model, input:torch.Tensor, class_mapping:Tuple[int])->int:
    model.eval()
    prob = model(input)
    prediction = int(prob>0.5)
    return prediction

class CNNInference:
    def __init__(self,class_mapping:Tuple[int]=[0, 1]) -> None:
        self.model_cnn = CNNetwork()
        self.state_dict = torch.load("state_dict_model.pt",map_location=torch.device('cpu'))
        self.model_cnn.load_state_dict(self.state_dict)
        self.class_mapping=class_mapping

        self.mel_spectrogram = T.MelSpectrogram(
       sample_rate=44100,
       n_fft=1024,
        win_length=None,
        hop_length=512,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm='slaney',
      onesided=True,
        n_mels=64,
        mel_scale="htk",
        ) 
        self.audio_transform = T.MelSpectrogram(
       sample_rate=44100,
       n_fft=1024,
        win_length=None,
        hop_length=512,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm='slaney',
      onesided=True,
        n_mels=64,
        mel_scale="htk",
        ) 
    

    def get_prediction(self,x:torch.Tensor)->int:

        
        
        print(x.size())
        x= F.resample(x, 44100, 16000)
        print(x.size())

        # if x.shape[1] > 16000:
        #     x = x[:, :16000]
        # elif x.shape[1] < 16000:
        #     num_missing_samples = 16000-x.shape[1]
        #     last_dim_padding = (0, num_missing_samples)
        #     x = torch.nn.functional.pad(x, last_dim_padding)
        mel_spectro=self.mel_spectrogram(x)
        print(mel_spectro.size())
        mel_spectro = mel_spectro.unsqueeze(0)
    
        print(mel_spectro)
        print(mel_spectro.unsqueeze(0).size())
        prediction =predict(self.model_cnn,mel_spectro.unsqueeze(0),self.class_mapping)
        return prediction




# x = torchaudio.load('15.wav')
# start = time.time()
# inference = CNNInference()
# print(inference.get_prediction(x[0]))
# end = time.time()
# print(end-start)