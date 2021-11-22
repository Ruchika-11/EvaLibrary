import torch
import torch.nn.functional as F
from EvaLibrary.src.dataloader import testloader
import torch.nn as nn 


def test(model, device, test_loader):

    model.eval()
    test_loss = 0
    correct = 0
    test_losses = []
    test_acc = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target,reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Test Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100*correct/len(test_loader.dataset))
    
    return test_losses,test_acc