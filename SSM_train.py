
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from scipy import stats
import numpy as np
import torchvision
from timeit import default_timer as timer
import argparse
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from early_stopping import EarlyStopping

from ssm import SSM
from mgnet import MgNet
from resnet import ResNet
from resnet import BasicBlock

    
use_cuda = torch.cuda.is_available()
print('Use GPU?', use_cuda)

###args
def get_args():
    parser = argparse.ArgumentParser(description='A general training enviroment for different learning methods.')

    parser.add_argument('--cuda',  action='store_true', help='use cuda')

    # For output 
    parser.add_argument('--name', default='result', type=str, help='the path to save the result')

    # For trail control
    parser.add_argument('--trail', default=1, type=int, help='trail of the same params.')
    
    # For methods    
    parser.add_argument('--epochs', type=int, help='epoch number', default=120)
    
    parser.add_argument('--batchsize', default=128, type=int, metavar='N', help='mini_batch size (default: 128)')
    
    parser.add_argument('--lr', '--learning-rate', default=1.0, type=float, metavar='LR', help='initial learning rate')
    
    parser.add_argument('--drop', default = 10, type=int, help='learning rate drop factor')
    
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    
    parser.add_argument('--dampening', default=0.9, type=float, metavar='D', help='dampening')
    
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (e.g. 5e-4)')
            
    parser.add_argument('--trun', metavar='truncate', default=0.02, type=float,  help='truncate value (default: 0.02)')
    
    parser.add_argument('--sig', metavar='significance', default=0.05, type=float,  help='significance level (default: 0.05)')
    
    parser.add_argument('--lk', metavar='leak ratio', default=8, type=int,  help='leak ratio (default: 8)')

    parser.add_argument('--minstat', metavar='minstat', default=100, type=int, help='mini-stat (default: 100)')
    
    parser.add_argument('--samplefreq', '--sf', metavar='samplefreq', default=10, type=int, help='sampling frequency (default: 10)')

    parser.add_argument('--varmode', '--vm', metavar='variance_mode', default="bm", type=str, help='variance mode (default: bm)')
    
    parser.add_argument('--keymode', '--km', metavar='key_mode', default="loss_plus_smooth", type=str, help='key mode (default: loss_plus_smooth')
    
    
    # For Data    
    parser.add_argument('--data', type=str, help='cifar10, cifar100 or mnist', default='cifar10')
    
    # For models    
    parser.add_argument('--model', type=str, help='mgnet128, resnet34 or preresnet18', default='mgnet128')
    
    return parser.parse_args()


def main():
    args = get_args()  # get the arguments

    #implementation
    minibatch_size = args.batchsize
    num_epochs =  args.epochs
    if args.data == 'cifar10':
        num_classes = 10
        num_channel_input = 3 
    elif args.data == 'cifar100':
        num_classes = 100
        num_channel_input = 3 

    
    # Step 1: Define a model
    if args.model == 'mgnet128':
        my_model = MgNet(num_channel_input, [2,2,2,2], 128, 128, num_classes)
    elif args.model  == 'mgnet256':
        my_model = MgNet(num_channel_input, [2,2,2,2], 256, 256, num_classes)
    elif args.model  == 'resnet18':
        my_model = ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes)
    elif args.model  == 'resnet34':
        my_model = ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)
    '''
    elif args.model  == 'preactresnet18':
        my_model = PreActResNet18()
    elif args.model  == 'preactresnet34':
        my_model = PreActResNet34()
    elif args.model  == 'densenet121':
        my_model = models.densenet121()
    elif args.model  == 'densenet161':
        my_model = models.densenet161()
    elif args.model  == 'efficientnet':
        my_model = EfficientNet.from_pretrained('efficientnet-b0')
    '''

    if use_cuda:
        my_model = my_model.cuda()
    
    # Step 2: Define a loss function and training algorithm
    criterion = nn.CrossEntropyLoss()
    
    
    # Step 3: load dataset
    if args.data == 'cifar10':
        normalize = torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        transform_train = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(32, padding=4),
                                                          torchvision.transforms.RandomHorizontalFlip(),
                                                          torchvision.transforms.ToTensor(),
                                                          normalize])
        transform_test  = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),normalize])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    if args.data == 'cifar100':
        normalize = torchvision.transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
        transform_train = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(32, padding=4),
                                                          torchvision.transforms.RandomHorizontalFlip(),
                                                          torchvision.transforms.ToTensor(),
                                                          normalize])
        transform_test  = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),normalize])
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)


    # obtain training indices that will be used for validation
    valid_size = 0.2    # percentage of training set to use as validation
    patience = 7
    delta  = 0.0005
    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    
    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
        
    # load validation data in batches
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=minibatch_size, sampler=train_sampler ,num_workers=0)        
    valid_loader = torch.utils.data.DataLoader(trainset, batch_size=minibatch_size, sampler=valid_sampler, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=minibatch_size, shuffle=False)
    
    # define the optimizer
    optimizer = SSM(my_model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, dampening=args.dampening, testfreq=len(trainloader), var_mode=args.varmode, leak_ratio=args.lk, minN_stats=args.minstat, mode=args.keymode, samplefreq=args.samplefreq, significance=args.sig, drop_factor=args.drop, trun=args.trun)

    test_accuracy_list = []
    lr_list = []
    statistic_list = []
    avg_loss_list = []
    time_list = []
    valid_losses = []
    stopped_epoch = 0
    early_stopping = EarlyStopping(patience=patience, delta=delta , verbose=True)
    
    
    #Step 4: Train the NNs
    # One epoch is when an entire dataset is passed through the neural network only once.
    for epoch in range(num_epochs):
        start = timer()
        running_loss = 0
        my_model.train()
        for i, (images, labels) in enumerate(trainloader):
            if use_cuda:
              images = images.cuda()
              labels = labels.cuda()
    
            # Forward pass to get the loss
            if (args.model == "mgnet128") or (args.model == "mgnet256"):
                outputs = my_model(0,images)   # We need additional 0 input for u in MgNet
            else:
                outputs = my_model(images) 
            loss = criterion(outputs, labels)
            optimizer.state['loss'] = loss.item()
            # Backward and compute the gradient
            optimizer.zero_grad()
            loss.backward()  #backpropragation
            running_loss += loss.item()
            optimizer.step() #update the weights/parameters
        avg_loss_list.append(running_loss)
        
          
        # validate the model
        my_model.eval()
        for images, labels in valid_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            if (args.model == "mgnet128") or (args.model == "mgnet256"):
                outputs = my_model(0,images)   # We need additional 0 input for u in MgNet
            else:
                outputs = my_model(images) 
            # calculate the loss
            loss = criterion(outputs, labels)
            # record validation loss
            valid_losses.append(loss.item())
        avg_valid_loss = np.average(valid_losses)
        valid_losses = []
        early_stopping(avg_valid_loss, my_model)
        
        
        # Training accuracy
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(trainloader):
            with torch.no_grad():
              if use_cuda:
                  images = images.cuda()
                  labels = labels.cuda()  
              if (args.model == "mgnet128") or (args.model == "mgnet256"):
                 outputs = my_model(0,images)   # We need additional 0 input for u in MgNet
              else:
                 outputs = my_model(images) 
              p_max, predicted = torch.max(outputs, 1) 
              total += labels.size(0)
              correct += (predicted == labels).sum()
        training_accuracy = float(correct)/total        
        
        # Test accuracy
        my_model.eval()
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(testloader):
            with torch.no_grad():
              if use_cuda:
                  images = images.cuda()
                  labels = labels.cuda()
              if (args.model == "mgnet128") or (args.model == "mgnet256"):
                  outputs = my_model(0,images)   # We need additional 0 input for u in MgNet
              else:
                  outputs = my_model(images) 
              p_max, predicted = torch.max(outputs, 1) 
              total += labels.size(0)
              correct += (predicted == labels).sum()
        test_accuracy = float(correct)/total
        
        end = timer()
        
 
        test_accuracy_list.append(test_accuracy)
        statistic_list.append(optimizer.state['statistic'])
        lr_list.append(optimizer.state['lr'])
        time_list.append(end - start)
        
        if early_stopping.early_stop:
            stopped_epoch = epoch
            break
    
    if args.trun == 0.02:
        sign_trun = '002'
    elif args.trun == 0.03:
        sign_trun = '003'
    if args.dampening == 0.9:
        sign_dampening = '09'
    elif args.dampening == 0.0:
        sign_dampening = '00'
    if args.lr == 1:
        sign_lr = '10'
    elif args.lr == 0.1:
        sign_lr = '01'
    if args.weight_decay == 0.0001:
        sign_wd = '00001'
    elif args.weight_decay == 0.0005:
        sign_wd = '00005'
    loglr_list = list(np.log10(np.array(lr_list)))
    
   
    
    print('complete')
    # example of files for training  
    f = open("SSM_train_data", 'a')
    """
    f.write('{}_trun{}_lk{}_sf{}_d{}_testaccu = {}\n'.format(args.model, sign_trun, args.lk, args.samplefreq, sign_dampening, test_accuracy_list))
    f.write('{}_trun{}_lk{}_sf{}_d{}_loglr = {}\n'.format(args.model, sign_trun, args.lk, args.samplefreq, sign_dampening, loglr_list))
    f.write('{}_trun{}_lk{}_sf{}_d{}_stat = {}\n'.format(args.model, sign_trun, args.lk, args.samplefreq, sign_dampening, statistic_list))
    f.write('{}_trun{}_lk{}_sf{}_d{}_loss = {}\n'.format(args.model, sign_trun, args.lk, args.samplefreq, sign_dampening, avg_loss_list))
    f.write('{}_trun{}_lk{}_sf{}_d{}_time = {}\n'.format(args.model, sign_trun, args.lk, args.samplefreq, sign_dampening, time_list))
    f.write("\n")
    """
    
    """
    f.write('ssm_{}_lr{}_wd{}_data{}_testaccu = {}\n'.format(args.model, sign_lr, sign_wd, args.data, test_accuracy_list))
    f.write('ssm_{}_lr{}_wd{}_data{}_loglr = {}\n'.format(args.model, sign_lr, sign_wd, args.data, loglr_list))
    f.write('ssm_{}_lr{}_wd{}_data{}_stat = {}\n'.format(args.model, sign_lr, sign_wd, args.data, statistic_list))
    f.write('ssm_{}_lr{}_wd{}_data{}_loss = {}\n'.format(args.model, sign_lr, sign_wd, args.data, avg_loss_list))
    f.write('ssm_{}_lr{}_wd{}_data{}_time = {}\n'.format(args.model, sign_lr, sign_wd, args.data, time_list))
    f.write('ssm_{}_lr{}_wd{}_data{}_stoppedepoch = {}\n'.format(args.model, sign_lr, sign_wd, args.data, stopped_epoch))
    f.write("\n")
    f.close()
    """
    f.write("lr{}_testaccu = {}\n".format(sign_lr,test_accuracy_list))
    f.write("lr{}_loglr = {}\n".format(sign_lr,loglr_list))
    f.write("lr{}_stopped_epoch = {}\n".format(sign_lr, stopped_epoch))
    f.write("\n")
    
    

main()



