
import torch
import torchaudio

class PreprocessingData():

          @staticmethod

          
          def Resampling(signal, sample_rate, new_sample_rate):
            #dat = []
            #dat.append([all_data[i][0],F.resample(all_data[i][1][0],all_data[i][1][1], sample_rate)])
              signal = torchaudio.functional.resample(signal,sample_rate, new_sample_rate)
              return signal

class ProcessingData():
          
          @staticmethod
          def MelSpec(signal, audio_transform,audio_transform_augm,Augm):
           # melspectrograms=[]
            #for i in range(len(proper_data)):
            if Augm:
              melspec = audio_transform_augm(signal)
            else:
                # melspectrograms.append(audio_transform(proper_data[i][1]).squeeze(0).transpose(0,1))
              melspec = audio_transform(signal)#.squeeze(0)#.transpose(0,1)
            #melspec = nn.utils.rnn.pad_sequence(melspec, batch_first=True).unsqueeze(1).transpose(2, 3) PROBLEME
            #for i in range(len(melspectrograms)):
             # proper_data[i][1] = melspectrograms[i]
            return melspec
              #liste = list(all_data[i][1])
              #liste[0] = spectrograms[i]


class SpecAugment(torch.nn.Module):
    """Augmentation technique to add masking on the time or frequency domain"""

    def __init__(self, rate, policy=3, freq_mask=2, time_mask=4):
        super(SpecAugment, self).__init__()

        self.rate = rate

        self.specaug = torch.nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )

        self.specaug2 = torch.nn.Sequential(
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
        # probability = torch.rand(1, 1).item()
        # if self.rate > probability:
            return  self.specaug(x)
        # return x

    def policy2(self, x):
        # probability = torch.rand(1, 1).item()
        # if self.rate > probability:
            return  self.specaug2(x)
        # return x

    def policy3(self, x):
        probability = torch.rand(1, 1).item()
        if probability > 0.5:
            return self.policy1(x)
        return self.policy2(x)

