import torch
import torchaudio
import os
from ProcessData import SpecAugment, PreprocessingData, ProcessingData
import random



class CustomDataset():
  
  def __init__(self, path_dir1,path_dir2,audio_transformation,sample_rate):   
    self.sample_rate = sample_rate 
    self.all_data = []
    self._build(path_dir1, path_dir2)
    self.noises = []
    for i in os.listdir("NOISES"):
          noise,_ = torchaudio.load("NOISES/"+i)
          self.noises.append(noise)

    if (audio_transformation == "melspec"):
      self.audio_transform = torchaudio.transforms.MelSpectrogram(
       sample_rate=sample_rate,
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
      self.audio_transform_augm = torch.nn.Sequential(
       torchaudio.transforms.MelSpectrogram(
       sample_rate=sample_rate,
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
        ), 
        SpecAugment(rate=0.5) 
      )
      
          
    
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
          #self.all_data.append(tup)



  def __getitem__(self, num):

    signal, label = self.all_data[num][1][0], self.all_data[num][0]
    if (self.all_data[num][1][1] != self.sample_rate):
      signal = PreprocessingData.Resampling(signal, self.all_data[num][1][1], self.sample_rate)
    #data = PreprocessingData.Proper(data)

    if (num%3==0)and(num<self.sizeHotwords):

       signal = ProcessingData.MelSpec(signal, self.audio_transform, self.audio_transform_augm, True)
    elif (num%3==2)and(num<self.sizeHotwords): #or(num%4==3))

            bruit = self.noises[random.randint(0,len(self.noises)-1)]
            bruit = torchaudio.transforms.Resample(44100, self.sample_rate)(bruit)
            speech_power = signal.norm(p=2)
            bruit_power = bruit.norm(p=2)          
            snr_dbs = [6, 3, 1] #change noise strength 20 10 3
            snr = 10 ** (snr_dbs[random.randint(0,len(snr_dbs)-1)] / 20)
            scale = snr * bruit_power / speech_power
            signal = (scale * signal + bruit) / 2
            signal = ProcessingData.MelSpec(signal, self.audio_transform, self.audio_transform_augm,False)
         


    else :
       signal = ProcessingData.MelSpec(signal, self.audio_transform, self.audio_transform_augm,False)
    #label, signal = self.all_data[num]
    return label, signal
    

   # label = self.all_data[num][0]
   # singlefile = self.all_data[num][1]
   # signal = list(singlefile)[0]
   # 
   # return label, signal 


  def __len__(self):
    return len(self.all_data)