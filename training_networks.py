#%%
# load dataloader
#%env mps_LAUNCH_BLOCKING=1
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from scripts.pGesture_Dataset import pGesture_Dataset
from models import CyclicLR

import random
from pytictoc import TicToc
from time import time

def cosine(t_max, eta_min=0):
    
    def scheduler(epoch, base_lr):
        t = epoch % t_max
        return eta_min + (base_lr - eta_min)*(1 + np.cos(np.pi*t/t_max))/2
    
    return scheduler
    
def training_networks(net, criterion, optimizer, filename, lr = 0.001, epochs = 1000, shuffle_flag = False): 
    random_seed = 0
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    batch_size = 64
    
    train_dataset = pGesture_Dataset("annotations/annotations_training.csv","./data/training_data", 
                                     transform = transform)
    test_dataset = pGesture_Dataset("annotations/annotations_testing.csv","./data/testing_data", 
                                     transform = transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_flag)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_flag)
    
    n_epochs = epochs
    iterations_per_epoch = len(train_dataset)
    best_accuracy = 0
    patience, trials = 100, 0
    
    optimizer.lr = lr
    sched = CyclicLR(optimizer, cosine(t_max=iterations_per_epoch * 3, eta_min=lr/100))
    
    cost_time = TicToc()
    computed_time0 = time()
    
    print('Start model training')
    lr_text = open('learned_model/log/learning_rate_'+ filename +'.txt', 'w')
    
    cost_time.tic()
    for epoch in range(1, n_epochs + 1):   # 데이터셋을 수차례 반복합니다.
        net.train()
        
        for idx, data in enumerate(train_dataloader):
            inputs_train = data['dataset'].reshape(-1, 100, net.input_dim).float().to("mps")
            labels_train = data['label'].to("mps")
    
            optimizer.zero_grad()
    
            outputs_train = net(inputs_train)
    
            loss = criterion(outputs_train, labels_train)
            loss.backward()
            
            optimizer.step()
            sched.step()
    
        net.eval()
        correct, total = 0, 0
        
        for idx, data in enumerate(test_dataloader):
            inputs_test = data['dataset'].reshape(-1, 100, net.input_dim).float().to("mps")
            labels_test = data['label'].to("mps")
            
            outputs_test = net(inputs_test)
            
            preds = F.log_softmax(outputs_test, dim=1).argmax(dim=1)
            total += labels_test.size(0)
            
            correct += (preds == labels_test).sum().item()
        
        accuracy = correct / total
        
        lr_text.write(str(epoch)+'\t')
        lr_text.write(str(loss.item())+'\t')
        lr_text.write(str(accuracy)+'\t')
        lr_text.write(str(best_accuracy)+'\t')
        lr_text.write(str(correct)+'\t')
        lr_text.write(str(total)+'\n')
        lr_text.write(str(time() - computed_time0)+'\n')
    
        if epoch % 5 == 0:
            print(f'Epoch: {epoch:3d}. Loss: {loss.item():.4f}. Acc.: {accuracy:2.2%}')
    
        if accuracy > best_accuracy:
            trials = 0
            best_accuracy = accuracy
            PATH = './learned_model/model_'+filename+'.pth'
            torch.save(net.state_dict(), PATH)
            print(f'Epoch {epoch} best model saved with accuracy: {best_accuracy:2.2%}')
        else:
            trials += 1
            if trials >= patience:
                print(f'Early stopping on epoch {epoch}')
                break
                
    cost_time.toc()
    lr_text.close()
