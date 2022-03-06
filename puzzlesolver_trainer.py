import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision

from torchvision import transforms
import time
import utils
import random
import numpy

trainfile = '/home/labs/testing/class65'
import os
import sys
sys.path.append(os.path.dirname(os.path.expanduser(trainfile)))
from vision_transformer import *
from JigsawImageLoader import *
#from puzzleLoss import *


def train_epoch(model, optimizer, data_loader, loss_history, device,criterion,num_mask):
  total_samples = len(data_loader.dataset)
  model.train()
  for i, (data) in enumerate(data_loader):
    data = data.to(device)
    batch_data=data.shape[0]
    #num_patches=model.patch_embed.num_patches
    mask,labels=produceMask(num_patches=196,num_masked=num_mask,batch_Size=batch_data)
    mask=mask.to(device)
    labels=labels.to(device)
    #print(mask)
    optimizer.zero_grad()
    output = model(data,mask,labels)
    #print(output.shape)
    #print(labels.shape)

    loss = criterion(output.permute(0,2,1),labels)
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
      print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
            ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
            '{:6.4f}'.format(loss.item()))
      loss_history.append(loss.item())


def evaluate(model, data_loader, loss_history, device,num_mask):
  model.eval()
  
  total_samples = len(data_loader.dataset)
  correct_samples = 0
  total_loss = 0

  with torch.no_grad():
    for data in data_loader:
      data = data.to(device)
      batch_data=data.shape[0]
      mask,labels=produceMask(num_patches=196,num_masked=num_mask,batch_Size=batch_data)
      mask=mask.to(device)
      labels=labels.to(device)
      output = model(data,mask,labels)
      loss_func=nn.CrossEntropyLoss()
      loss = loss_func(output.permute(0,2,1),labels)
      _, pred = torch.max(output.permute(0,2,1), dim=1)
      
      total_loss += loss.item()
      correct_samples += pred.eq(labels).sum()

  avg_loss = total_loss / total_samples
  loss_history.append(avg_loss)
  
  print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
       '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
        '{:5}'.format(total_samples*num_mask) + ' (' +
        '{:4.2f}'.format(100.0 * correct_samples / (total_samples*num_mask)) + '%)\n')


def train_vit(loadCheck=False):
  # set random seed
  seed=42
  torch.manual_seed(seed)
  #random.seed(seed)
  
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  start_time = time.time()

  BATCH_SIZE_TRAIN = 256
  BATCH_SIZE_TEST = 1000
  N_EPOCHS = 1
  n_workers=10
  
  # set same seed for experiments
  g=torch.Generator()
  g.manual_seed(seed)
  
  trainpath = '/shareDB/imagenet/train'
  #trainpath = '/ILSVRC2012_img_train'
    #if os.path.exists(trainpath+'_255x255'):
    #    trainpath += '_255x255'
  #train_set = torchvision.datasets.ImageNet(root='/shareDB/imagenet', split='train',loader=DataLoader)
  train_set = DataLoader(data_path=trainpath, txt_list='text_file_imagenet_training_images.txt', classes=1000)
  #train_set = DataLoader(trainpath,classes=1000)
  train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=n_workers,worker_init_fn=seed_worker,generator=g)
    
    #if os.path.exists(valpath+'_255x255'):
    #    valpath += '_255x255'
  test_set = DataLoader(data_path=trainpath, txt_list='text_file_imagenet_training_images.txt',classes=1000)
  test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=n_workers,worker_init_fn=seed_worker,generator=g)
    # N = train_set.N
    
    # iter_per_epoch = train_set.N/args.batch
    # print('Images: train %d, validation %d'%(train_set.N,test_set.N))

  # Initialize model
  headLoc=locationHead(in_dim=768,num_patches=196)
  model=VisionTransformer()
  model= utils.MultiCropWrapper(model,headLoc)
  model = model.to(device)
  
  #criterion=puzzleLossCE()
  criterion =nn.CrossEntropyLoss()

  optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
  #print(torch.cuda.memory_summary(device=None, abbreviated=False))
  epoch_start=1  
  train_loss_history, test_loss_history = [], []
  if loadCheck==True:
    checkpoint=torch.load('checkpoint100.pth')
    epoch=checkpoint['epoch']
    model.load_state_dict(checkpoint['model_sd'])
    optimizer.load_state_dict(checkpoint['optimizer_sd'])
    criterion=checkpoint['criterion']
    train_loss_history=checkpoint['train_loss_history'] 
    test_loss_history=checkpoint['test_loss_history']
    epoch_start=epoch
    print(epoch)
  # Train model
  masked_history=[]
  for epoch in range(epoch_start, N_EPOCHS + epoch_start):
    curr_start_time = time.time()
    num_mask=196
    train_epoch(model, optimizer, train_loader, train_loss_history, device,criterion,num_mask)
    evaluate(model, test_loader, test_loss_history, device,num_mask)
    masked_history.append(num_mask)
    print(f'Epoch {epoch} execution time:', '{:5.2f}'.format((time.time() - curr_start_time) / 60), 'minutes\n')
    checkpoint=torch.save({'epoch': epoch, 'model_sd': model.state_dict(), 'optimizer_sd': optimizer.state_dict(), 'criterion': criterion,'train_loss_history':train_loss_history,'test_loss_history':test_loss_history},"checkpoint196.pth")

  print('Execution time:', '{:5.2f}'.format((time.time() - start_time) / 60), 'minutes')

def produceMask(num_patches=196,num_masked=0,batch_Size=1):
  ind_masked=torch.randint(0,num_patches-1,(batch_Size,num_masked,))
  mask=torch.zeros(batch_Size,num_patches,)
  mask[:,ind_masked]=1
  return mask,ind_masked

def seed_worker(worker_id):
  worker_seed=torch.initial_seed()% 2**32
  random.seed(worker_seed)
  numpy.random.seed(worker_seed)

