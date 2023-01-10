import torchaudio
from inference import CNNInference
import matplotlib.pyplot as plt 
import librosa


def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Melspectrogram (db)')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)
  plt.show()



signal,_ = torchaudio.load("float32.wav")
print(signal.size())

inference = CNNInference()
print(inference.get_prediction(signal))
plot_spectrogram(inference.get_prediction(signal).squeeze(0).squeeze(0))