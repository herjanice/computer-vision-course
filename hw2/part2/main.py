
import torch
import os


from torch.utils.data import DataLoader
import torch.optim as optim 
import torch.nn as nn

from myModels import  myLeNet, myResnet
from myDatasets import  get_cifar10_train_val_set
from tool import train, fixed_seed

# Modify config if you are conducting different models
# from cfg import LeNet_cfg as cfg
from cfg import Resnet_cfg as cfg

# from torchsummary import summary

def new_filename(path, filename, extension):
    counter = 1
    root_path = path
    while os.path.exists(path):
        path = os.path.join(root_path, filename+"_"+str(counter)+extension)
        counter += 1
    return path


def train_interface():
    
    """ input argumnet """

    data_root = cfg['data_root']
    model_type = cfg['model_type']
    num_out = cfg['num_out']
    num_epoch = cfg['num_epoch']
    split_ratio = cfg['split_ratio']
    seed = cfg['seed']

    print("epochs:", num_epoch)
    
    # fixed random seed
    fixed_seed(seed)
    

    # os.makedirs( os.path.join('./acc_log',  model_type), exist_ok=True)
    os.makedirs( os.path.join('./save_dir', model_type), exist_ok=True)    
    log_path = os.path.join('./acc_log', model_type, 'acc_' + model_type + '_.log')
    # log_path = new_filename('./acc_log/'+model_type, 'acc_'+model_type, '.log')
    save_path = os.path.join('./save_dir', model_type)

    # with open(log_path, 'w'):
    #     pass
    
    ## training setting ##
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu') 
    
    
    """ training hyperparameter """
    lr = cfg['lr']
    batch_size = cfg['batch_size']
    milestones = cfg['milestones']
    
    
    ## Modify here if you want to change your model ##
    # model = myLeNet(num_out=num_out)
    model = myResnet()
    # print model's architecture
    print(model)
    # summary(model, (3,32,32))

    # Get your training Data 
    # TODO: FINISHED
    # You need to define your cifar10_dataset yourself to get images and labels for each data
    # Check myDatasets.py 
      
    train_set, val_set =  get_cifar10_train_val_set(root=data_root, ratio=split_ratio)    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    # define your loss function and optimizer to unpdate the model's parameters.
    
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,weight_decay=1e-6, nesterov=True)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=milestones, gamma=0.1)
    
    # We often apply crossentropyloss for classification problem. Check it on pytorch if interested
    criterion = nn.CrossEntropyLoss()
    
    # Put model's parameters on your device
    model = model.to(device)
    
    # TODO: FINISHED 
    # Complete the function train
    # Check tool.py
    train(model=model, train_loader=train_loader, val_loader=val_loader, 
          num_epoch=num_epoch, log_path=log_path, save_path=save_path,
          device=device, criterion=criterion, optimizer=optimizer, scheduler=scheduler)

    
if __name__ == '__main__':
    train_interface()




    