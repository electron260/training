
#Recup de la data 


#Modif de la data 
import pyaudio
import random
import sys
import wave

from requests import session 
import sounddevice as sd
import os
from scipy.io.wavfile import write 
import numpy as np

# sample_rate = 16000
# filename = 'myfile.wav'

# # Set chunk size of 1024 samples per data frame
# chunk = 1024  

# # Open the sound file 
# wf = wave.open(filename, 'rb')

# # Create an interface to PortAudio
# p = pyaudio.PyAudio()

# # Open a .Stream object to write the WAV file to
# # 'output = True' indicates that the sound will be played rather than recorded
# stream = p.open(format = p.get_format_from_width(wf.getsampwidth()),
#                 channels = wf.getnchannels(),
#                 rate = wf.getframerate(),
#                 output = True)

# # Read data in chunks
# data = wf.readframes(chunk)

# # Play the sound by writing the audio data to the stream
# while data != '':
#     stream.write(data)
#     data = wf.readframes(chunk)

# # Close and terminate the stream
# stream.close()
# p.terminate()


#Recupération de la data 


#### IMPORTS ####################



def record_audio_and_save(n_times:int, session_name:str ,Kw:bool):
    #input("To start recording Wake Word press Enter: ")
    for i in range(n_times):
        fs = 44100
        seconds = 2
        sd.default.device = 11
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()
        if Kw :
            write("Hotword-"+session_name+"-"+str(i) + ".wav", fs, myrecording)
        else:
            write("Noise-"+session_name+"-"+str(i) + ".wav", fs, myrecording)
        #input(f"Press to record next or two stop press ctrl + C ({i + 1}/{n_times}): ")
record_audio_and_save(150 , "NoiseDVIC3", False)