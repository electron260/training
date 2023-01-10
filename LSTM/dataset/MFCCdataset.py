from numpy import int16
import torchaudio
import torch
import torch.nn as nn
import librosa 
import numpy as np
from sonopy import power_spec, mel_spec, mfcc_spec, filterbanks
import os
import torch.utils.data as data
import matplotlib.pyplot as plt
from matplotlib import cm
import random


class MFCC(nn.Module):

    def __init__(self, sample_rate, fft_size=400, window_stride=(400, 200), num_filt=40, num_coeffs=40):
        super(MFCC, self).__init__()
        self.sample_rate = sample_rate
        self.window_stride = window_stride
        self.fft_size = fft_size
        self.num_filt = num_filt
        self.num_coeffs = num_coeffs
        self.mfcc = lambda x: mfcc_spec(
            x, self.sample_rate, self.window_stride,
            self.fft_size, self.num_filt, self.num_coeffs
        )
    
    def forward(self, x):
        return torch.Tensor(self.mfcc(x.squeeze(0).numpy())).transpose(0, 1).unsqueeze(0)



def get_featurizer(sample_rate):
    return MFCC(sample_rate=sample_rate)



class SpecAugment(nn.Module):
    """Augmentation technique to add masking on the time or frequency domain"""

    def __init__(self, rate, policy=3, freq_mask=2, time_mask=4):
        super(SpecAugment, self).__init__()

        self.rate = rate

        self.specaug = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )

        self.specaug2 = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )

        policies = { 1: self.policy1, 2: self.policy2, 3: self.policy3 }
        self._forward = policies[policy]

    def forward(self, x):
        return self._forward(x)

    def policy1(self, x):
        probability = torch.rand(1, 1).item()
        if self.rate > probability:
            return  self.specaug(x)
        return x

    def policy2(self, x):
        probability = torch.rand(1, 1).item()
        if self.rate > probability:
            return  self.specaug2(x)
        return x

    def policy3(self, x):
        probability = torch.rand(1, 1).item()
        if probability > 0.5:
            return self.policy1(x)
        return self.policy2(x)




class WakeWordData(torch.utils.data.Dataset):
    """Load and process wakeword data"""

    def __init__(self,path_dir1, path_dir2, sample_rate=16000, valid=False):
        self.sr = sample_rate
        self.all_data =[]
        self._build(path_dir1,path_dir2)  
        self.audio_transform = get_featurizer(sample_rate)
        self.audio_transform_augm = nn.Sequential(
                get_featurizer(sample_rate),
                SpecAugment(rate=0.5)
            )      
        self.noises = []
        for i in os.listdir("NOISES"):
            noise,_ = torchaudio.load("NOISES/"+i)
            self.noises.append(noise)



    def _build(self, path_dir1, path_dir2):
    
        data_path_dic   = {
        0: [path_dir1+ i for i in os.listdir(path_dir1)],  #fichier oui
        1:[path_dir2+ i for i in os.listdir(path_dir2)] #fichier non
                      }
        self.sizeHotwords = 3*len(data_path_dic[0]) 
        for classi, path in data_path_dic.items():
            for single_file in path:
                audio = torchaudio.load(single_file)
                tup = classi, audio
                self.all_data.append(tup)
                if classi == 0:
                    self.all_data.append(tup)
                    self.all_data.append(tup)
                
                


    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.item()

    
        signal, label, sr = self.all_data[idx][1][0], self.all_data[idx][0],self.all_data[idx][1][1]
        
        if sr > self.sr:
            #print("audio transformed : ",sr," -----> ",self.sr)
            signal = torchaudio.transforms.Resample(sr, self.sr)(signal)
         
        if (idx%3==0)and(idx<self.sizeHotwords):
            mfcc = self.audio_transform_augm(signal)

        elif (idx%3==1)and(idx<self.sizeHotwords):
            bruit = self.noises[random.randint(0,len(self.noises)-1)]
            bruit = torchaudio.transforms.Resample(44100, self.sr)(bruit)
            speech_power = signal.norm(p=2)
            bruit_power = bruit.norm(p=2)          
            snr_dbs = [10, 5, 3] # 20 10 3 
            noisy_speeches = []
            snr = 10 ** (snr_dbs[random.randint(0,len(snr_dbs)-1)] / 20)
            scale = snr * bruit_power / speech_power
            signal = (scale * signal + bruit) / 2
            mfcc = self.audio_transform(signal)


        else :
            mfcc = self.audio_transform(signal)
        mfcc = mfcc.squeeze(0).transpose(0,1)
     
        return mfcc, label


if __name__ == "__main__":


    datatrain = WakeWordData("train/Hotword/","train/Noise/",16000)

    print(len(datatrain))

    train_dataloader = data.DataLoader(
    datatrain,
    batch_size=3,
    shuffle=True,    # shuffle tells us to randomly sample the
                     # dataset without replacement
    num_workers=4    # num workers is the number of worker processes
                     # that load from dataset in parallel while your
                     # model is processing stuff
    )

# for i,v in train_dataloader:
#   print(v.size())
#   print(i.size())



    def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
        fig, axs = plt.subplots(1, 1)
        axs.set_title(title or 'Spectrogram (db)')
        axs.set_ylabel(ylabel)
        axs.set_xlabel('frame')
        im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
        if xmax:
            axs.set_xlim((0, xmax))
        fig.colorbar(im, ax=axs)
        plt.show()


    print(datatrain.__len__())
    for i in range(30):
    #print(test.__getitem__(random.randint(0,200)))
        croute = datatrain.__getitem__(i)
        #croute = list(croute)
        #print(croute[0])
        print(croute[0].size())

        print(croute[1])
        
