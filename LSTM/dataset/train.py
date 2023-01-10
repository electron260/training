#import wandb
import torchaudio
import torch
from torch import nn
import torchaudio.functional as F
import os 
import librosa
from os.path import isfile, join 
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import wandb
#Recup de la data 


#Modif de la data 
import random
import sys
import numpy as np

#Custom dataset
from MFCCdataset import WakeWordData
#Model
from LSTMmodel import LSTMWakeWord

from torchsummary import summary


# seed = 123
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# np.random.seed(seed)
# random.seed(seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True





train_dataset = WakeWordData("train/Hotword/","train/Noise/",sample_rate=16000)
test_dataset = WakeWordData("test/Hotword/","test/Noise/",sample_rate=16000)
print("Size dataset train :")
print(len(train_dataset))
print("Size dataset test :")
print(len(test_dataset))

train_dataloader = data.DataLoader(
    train_dataset,
    batch_size=3,
    shuffle=True,    # shuffle tells us to randomly sample the
                     # dataset without replacement
    num_workers=3    # num workers is the number of worker processes
                     # that load from dataset in parallel while your
                     # model is processing stuff
)

test_dataloader = data.DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,    # shuffle tells us to randomly sample the
                     # dataset without replacement
    num_workers=4    # num workers is the number of worker processes
                     # that load from dataset in parallel while your
                     # model is processing stuff
)






def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Spectrogram (db)')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)
  plt.show(block=False)

# print(test_dataset.__len__())
# for i in range (test_dataset.__len__()):
#   #print(test.__getitem__(random.randint(0,200)))
#   croute = test_dataset.__getitem__(i)
#   croute = list(croute)

#   #print(croute[1].size())
#   print(croute[0].size())
#   #plot_spectrogram(croute[0].squeeze(0))

# for i in range (train_dataset.__len__()):
#   #print(test.__getitem__(random.randint(0,200)))
#   croute = train_dataset.__getitem__(i)
#   croute = list(croute)

#   #print(croute[1].size())
 # print(croute[0].size())


def binary_accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(preds)
    acc = rounded_preds.eq(y.view_as(rounded_preds)).sum().item() / len(y)
    return acc


best_acc = 0.0

def step(model, device, dataloader, dataset,  criterion, optimizer, train, epoch):
    model.train() if train else model.eval()
    data_len = len(dataloader)
    global best_acc
    # for epoch in tqdm(range(epochs), desc="Epoch"):
    #    with tqdm(train_dataloader, desc="Train") as pbar:

    labelstot = []
    preds = []
    losses = []
    accs = []
    #with experiment.train():
    if train:
       desc="train" 
    else :
       desc = "test"
    with tqdm( dataloader, desc) as pbar:
      for melspec, labels in pbar:

            melspec, labels = melspec.to(device), labels.to(device)         
            melspec = melspec.transpose(0,1)

            labels_estim = model(melspec)
    
            #Get the predictions : 
            pred = torch.sigmoid(labels_estim)
            preds += torch.flatten(torch.round(pred).to(device))       
            labelstot += torch.flatten(labels).to(device)

            if train :
              optimizer.zero_grad()
              
              loss= criterion(torch.flatten(labels_estim), labels.float())
              
              loss.backward()
              optimizer.step()
              losses.append(loss.item())
 
            if not train :
              acc = binary_accuracy(pred, labels)
              accs.append(acc)
             
            #  print(f"{total_loss:.2e}", f"{total_acc * 100:.2f}%")       
            #print(total_loss, total_acc)
            #pbar.set_postfix(loss=f"{total_loss:.2e}", acc=f"{acc * 100:.2f}%")
      if train :
        avg_train_loss = sum(losses)/len(losses)
        acc = binary_accuracy(torch.Tensor(preds), torch.Tensor(labelstot))
        print('avg train loss:', avg_train_loss, "avg train acc", acc)
        
        wandb.log({"avg train loss:": avg_train_loss, "epoch" : epoch})
        wandb.log({"avg train acc": acc, "epoch" : epoch})
 
      else :
        average_acc = sum(accs)/len(accs) 
        print('Average test Accuracy:', average_acc, "\n")
        if average_acc > best_acc :
                torch.save(model.state_dict(),"state_dict_model.pt")
                best_acc = average_acc
                print("--------------")
                print(best_acc)
                print("--------------")

     
        wandb.log({"average_acc": average_acc, "epoch" : epoch})




#model = WakeUpWord().cuda()

def main(device:str,epochs):


        
  # wandb.init(project=session_name, entity = "electron260")
  # wandb.run.name = run_name
  # wandb.config = {
  # "learning_rate": 0.001,
  # "epochs": epochs,
  # "batch_size": 5
  #   }
  model = LSTMWakeWord().to(device)
  #criterion = nn.CrossEntropyLoss()#.cuda()
  criterion = nn.BCEWithLogitsLoss().to(device)
  minlr = 10e-5
  lambda1 = lambda epoch: max(0.95 ** epoch, minlr / 10e-3)

  opti = optim.AdamW(model.parameters(), lr = 0.001)
  scheduler = torch.optim.lr_scheduler.LambdaLR(opti, lr_lambda=lambda1)
  for epoch in tqdm(range(epochs), desc="Epoch"):
   
  #   train(model,"cuda", dataloader, criterion, opti,  epoch = i+1)#scheduler ? 
        print("-----train-----")
        step(model,device, train_dataloader, train_dataset, criterion, opti,  True, epoch)
        print("-----test-----")
        step(model,device, test_dataloader, test_dataset, criterion, opti, False, epoch)
        print("--------------")
        print('Epoch:', epoch,'LR:', opti.param_groups[0]["lr"])
        scheduler.step()






epochs = 200

wandb.init(project="LSTM_new_dat_2", entity = "electron260")
wandb.run.name = "LSTM"
wandb.config = {
  "learning_rate": 0.001,
  "epochs": epochs,  
  "batch_size": 3
    }

print("debut main")


main("cuda", epochs)