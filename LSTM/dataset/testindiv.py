from numpy import int16
import torchaudio
import torch
import torch.nn as nn
import librosa.display
import numpy as np
from sonopy import power_spec, mel_spec, mfcc_spec, filterbanks
import os
import torch.utils.data as data
import matplotlib.pyplot as plt
from matplotlib import cm



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

sig, _ = torchaudio.load("float32.wav")
print(sig)
print(sig.size())
print(_)
signal = torchaudio.transforms.Resample(_, 16000)(sig)

print(signal.size())
audio_transform = get_featurizer(16000)
mfcc=audio_transform(signal)
print(mfcc)
print(mfcc.size())

# def plot_mfcc(mfcc_data):
#     fig, ax = plt.subplots()
#     mfcc_data= np.swapaxes(mfcc_data, 0 ,1)
#     cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
#     ax.set_title('MFCC')

#     plt.show()

# librosa.display.specshow(mfcc, x_axis='time')
# plt.colorbar()
# plt.tight_layout()
# plt.title('mfcc')
# plt.show
def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'MFCC (db)')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)
  plt.show()


plot_spectrogram(mfcc.squeeze(0))


print(mfcc.squeeze(0).transpose(0,1).size())