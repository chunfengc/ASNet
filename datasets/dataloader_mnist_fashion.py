
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F

def load_mnist_fashion_test(batch_size=100,data_path='../datasets/Fashion_MNIST/', class_label = None): 
    dataset = datasets.FashionMNIST(root=data_path, train=False, transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
    if class_label != None:
        idx = dataset.targets==class_label
        dataset.targets = dataset.targets[idx]
        dataset.data = dataset.data[idx]
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

def load_mnist_fashion_train(batch_size=100, data_path='../datasets/Fashion_MNIST/',class_label = None,shuffle=True): 
    dataset = datasets.FashionMNIST(root=data_path,download=True, train=True, transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
    if class_label != None:
        idx = dataset.targets==class_label
        dataset.targets = dataset.targets[idx]
        dataset.data = dataset.data[idx]
    return torch.utils.data.DataLoader(dataset,  batch_size, shuffle=shuffle)

def train_and_save(model,device, train_loader,test_loader):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(1, 5 + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
    torch.save(model.state_dict(), '../model/MNIST/best_mnist_fashion_model.pth')
        
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    