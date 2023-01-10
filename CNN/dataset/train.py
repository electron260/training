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
from d2l import torch as d2l
from torch.optim.lr_scheduler import StepLR
#Recup de la data 


#Modif de la data 
import random
import sys
import numpy as np

#Custom dataset
from customdataset import CustomDataset
#Model
from model import CNNetwork

from torchsummary import summary


# seed = 123
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# np.random.seed(seed)
# random.seed(seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True





train_dataset = CustomDataset("train/Hotword/","train/Noise/",audio_transformation="melspec",sample_rate=16000)
test_dataset = CustomDataset("test/Hotword/","test/Noise/",audio_transformation="melspec",sample_rate=16000)
print("Size dataset train :")
print(len(train_dataset))
print("Size dataset test :")
print(len(test_dataset))

train_dataloader = data.DataLoader(
    train_dataset,
    batch_size=3,
    shuffle=True,    # shuffle tells us to randomly sample the
                     # dataset without replacement
    num_workers=4    # num workers is the number of worker processes
                     # that load from dataset in parallel while your
                     # model is processing stuff
)

test_dataloader = data.DataLoader(
    test_dataset,
    batch_size=3,
    shuffle=False,    # shuffle tells us to randomly sample the
                     # dataset without replacement
    num_workers=4    # num workers is the number of worker processes
                     # that load from dataset in parallel while your
                     # model is processing stuff
)
negtest=0
postest = 0
for i in test_dataset:
  if i[0]==0:
    postest +=1
  else :
    negtest+=1
print("test --> neg : ",negtest,"    pos : ",postest)


negtrain=0
postrain = 0
for i in train_dataset:
  if i[0]==0:
    postrain +=1
  else :
    negtrain+=1
print("train --> neg : ",negtrain,"    pos : ",postrain)


# def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
#   fig, axs = plt.subplots(1, 1)
#   axs.set_title(title or 'Melspectrogram (db)')
#   axs.set_ylabel(ylabel)
#   axs.set_xlabel('frame')
#   im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
#   if xmax:
#     axs.set_xlim((0, xmax))
#   fig.colorbar(im, ax=axs)
#   plt.show(block=False)

# print(train_dataset.__len__())
# for i in range (12):
#   #print(test.__getitem__(random.randint(0,200)))
#   croute = train_dataset.__getitem__(i)
#   croute = list(croute)
#   #print(croute[1].size())
#   #print(croute[0])
#   plot_spectrogram(croute[1].squeeze(0))





best_acc = 0.0

def step(model, device, dataloader, dataset,  criterion, optimizer, train, epoch):
    model.train() if train else model.eval()
    data_len = len(dataloader)
    global best_acc
    # for epoch in tqdm(range(epochs), desc="Epoch"):
    #    with tqdm(train_dataloader, desc="Train") as pbar:
    total_loss = 0.0
    total_acc = 0.0


    

    #with experiment.train():
    if train:
       desc="train" 
    else :
       desc = "test"
    with tqdm( dataloader, desc) as pbar:
      for labels, melspec in pbar:


            actuals = []
            predictions = []
            #spectrograms, labels, input_lengths, label_lengths = data 
            #melspec, label = data
            #melspec.transpose(0,1)
           
            melspec, labels = melspec.to(device), labels.to(device)



            labels_estim = model(melspec)

            # print(labels_estim.reshape(-1))
            # print(labels.float)
            # print(labels)
            # print(labels.size())    
            # print(labels_estim)
            # print(labels_estim.size())      
            # print(labels_estim.reshape(-1))
            # print(labels_estim.reshape(-1).size())
          

            acc  = int(((labels_estim.reshape(-1) > 0.5).long() == labels).sum()) #/ len(train_dataloader.dataset)

            # for confusion matrix

            




            #acc = (labels_estim.argmax(1)==labels).sum().item()
            #acc = (labels == labels_estim.argmax(dim=1)).sum()
            # output = model(melspec)  # (batch, time, n_class)
            # output = F.log_softmax(output, dim=2)
            # output = output.transpose(0, 1) # (time, batch, n_class)
            # print(labels_estim.reshape(-1).size())
            # print(labels.size())

            loss = criterion(labels_estim.reshape(-1), labels.float())

            if train :
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()
              

         

            #experiment.log_metric('loss', loss.item(), step=iter_meter.get())
            #experiment.log_metric('learning_rate', scheduler.get_lr(), step=iter_meter.get())
      

           
            #scheduler.step()
            #iter_meter.step()
            # if batch_idx % 100 == 0 or batch_idx == data_len:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(spectrograms), data_len,
            #         100. * batch_idx / len(train_loader), loss.item()))

                
            total_loss += loss.item() /  dataset.__len__()
            total_acc += acc / dataset.__len__()  

          

            if not train :
              if total_acc > best_acc :
                torch.save(model.state_dict(),"state_dict_model.pt")
                best_acc = total_acc
                print("--------------")
                print(best_acc)
                print("--------------")

            #  print(f"{total_loss:.2e}", f"{total_acc * 100:.2f}%")       
            #print(total_loss, total_acc)
            pbar.set_postfix(loss=f"{total_loss:.2e}", acc=f"{total_acc * 100:.2f}%")
      if train :
        wandb.log({"loss_train": total_loss, "epoch" : epoch})
        wandb.log({"acc_train": total_acc, "epoch" : epoch})
      else :
        wandb.log({"loss_test": total_loss,"epoch" : epoch})
        wandb.log({"acc_test": total_acc, "epoch" : epoch})

      



def main(device:str,epochs):


   
  model = CNNetwork().to(device)
  criterion = nn.BCELoss().to(device)
  

  opti = optim.AdamW(model.parameters(), lr = 10e-3)
  minlr = 10e-5
  lambda1 = lambda epoch: max(0.95 ** epoch, minlr / 10e-3)
  
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

wandb.init(project="withphonesamples", entity = "electron260")
wandb.run.name = "total_equality_3conv"
wandb.config = {
  "learning_rate": 10e-4,
  "epochs": epochs,
  "batch_size": 3
    }

print("debut main")
main("cpu", epochs)