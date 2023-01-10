import pyaudio
import wave
import numpy as np

chunk = 14700 # Record in chunks of 256 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channel = 1
fs = 44100  # Record at 44100 samples per second
seconds = 2



def record(filename:str,hotword:bool):
    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    print('Recording')

    stream = p.open(format=sample_format,
                channels=channel,
                rate=fs,
                frames_per_buffer=chunk,
                input=True, input_device_index=11)

    frames = []  # Initialize array to store frames

# Store data in chunks for 2 seconds
    for i in range(0, int(fs  / chunk * seconds)):
        data = stream.read(chunk)

        print(type(data))
        frames.append(data)


    print(len(b''.join(frames)))
    print(type(b''.join(frames)))
# Stop and close the stream 
    stream.stop_stream()
    stream.close()
# Terminate the PortAudio interface
    p.terminate()

    print(len(frames))
    print('Finished recording')

    print(p.get_sample_size(sample_format))
# Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channel)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()



def record_2sec(nb:int, hotword:bool):
    session_name = input("Name of the session : ")
    for i in range(nb):
        input("Press Enter to start a recording")
        if hotword :
            record("KWORD-"+session_name+" " +str(i), hotword)
        else :
            record("NOISE-"+session_name +" "+str(i), hotword)




record_2sec(25,hotword = True)