import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import pdb 
    
def trainloader_resnet(batch_size, data_path='../datasets/',num_class=10, class_label = None,isshuffle=True,iscrop=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])  
    
    if iscrop:
        transform=torchvision.transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize])
    else:
        transform=torchvision.transforms.Compose([transforms.ToTensor(),normalize])
        
    
    if num_class == 10:
        dataset = torchvision.datasets.CIFAR10(root=data_path+'CIFAR10/',
                 train=True,transform=transform, download=True)
    else:
        dataset = torchvision.datasets.CIFAR100(root=data_path+'CIFAR100/',
                 train=True,transform=transform, download=True)
        
    if class_label != None:
        dataset = dataset_index(dataset,class_label)
    data_loader = torch.utils.data.DataLoader(dataset,
                   batch_size=batch_size, shuffle=True, pin_memory=True)
    return data_loader
    
def testloader_resnet(batch_size,  data_path='../datasets/', num_class=10,class_label = None):
    if num_class == 10:
        dataset = datasets.CIFAR10(root=data_path+'CIFAR10/', 
           train=False,
           transform=torchvision.transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
           ]))
    else:
        dataset = datasets.CIFAR100(root=data_path+'CIFAR100/',
           train=False,
           transform=torchvision.transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
           ]))
        
    if class_label != None:
        dataset = dataset_index(dataset,class_label)
    test_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size, shuffle=False)
    return test_loader


 
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
]) 

from torch.utils.data import Dataset
class dataset_index(Dataset):
    def __init__(self, base_dataset, class_label=None):
        self.base_dataset = base_dataset
        if class_label is None:
            self.indexes = list(np.arange(len(base_dataset)))
        else:
            self.indexes = []
            for i in range(len(base_dataset)):
                (d, t) = base_dataset[i]
                if t == class_label:
                    self.indexes.append(i)
    def __len__(self):
        return len(self.indexes)
    def __getitem__(self, i):
        index_ori = self.indexes[i]
        return self.base_dataset[index_ori]
    

def trainloader_cifar_vgg19(train_batch,data_path='../datasets/', num_class=10,shuffle=True, class_label = None,iscrop=True):
    if iscrop:
        transform=transform_train
    else:
        transform=transform_test
    
    if num_class==10:
        dataset = datasets.CIFAR10(root=data_path+'CIFAR10/', train=True, download=False, transform=transform)
    else:
        dataset = datasets.CIFAR100(root=data_path+'CIFAR100/', train=True, download=False, transform=transform)
    if class_label != None:
        dataset = dataset_index(dataset,class_label)
    trainloader = data.DataLoader(dataset, batch_size=train_batch, shuffle=shuffle)
    print('Train dataset loaded')
    return trainloader

def testloader_cifar_vgg19(test_batch,data_path='../datasets/', num_class=10, class_label = None):
    if num_class == 10:
        dataset = datasets.CIFAR10(root=data_path+'CIFAR10/', train=False, download=False, transform=transform_test)
    else:
        dataset = datasets.CIFAR100(root=data_path+'CIFAR100/', train=False, download=False, transform=transform_test)
        
    dataset = dataset_index(dataset,class_label)
    testloader = data.DataLoader(dataset, batch_size=test_batch, shuffle=False)
    print('Test dataset loaded')
    
    return testloader

 

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
    
# mean = (0.4914, 0.4822, 0.4465)
# std = (0.2023, 0.1994, 0.2010)
# unorm = UnNormalize(mean,std)


# transform = transforms.Compose([
#     # you can add other transformations in this list
#     transforms.ToTensor()
# ])
# testset = torchvision.datasets.CIFAR10('/home/cfcui/datasets/CIFAR10/', train=False, download=False,transform=transform)

# testloader = data.DataLoader(testset, batch_size=8, shuffle=False)
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# dataiter = iter(testloader)
# images, labels = dataiter.next()

# # print images
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(8)))

