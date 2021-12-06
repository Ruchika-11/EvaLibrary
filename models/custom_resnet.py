'''Custom ResNet in PyTorch for CIFAR10'''

# Session 9 ResNets and Higher Receptive Fields


import torch.nn as nn
import torch.nn.functional as F


class CustomResnet(nn.Module):
    def __init__(self):
        super(CustomResnet, self).__init__()

	#prep layer 
        self.preplayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False), 
            nn.BatchNorm2d(64),
            nn.ReLU(),  
            )
	
	#layer x in layer1
        self.x1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False), 
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),            
            )

	#Residual Block 1 
        self.r1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False), 
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            )

	#layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False), 
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),        
            )

	#layer x in layer 3
        self.x3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, bias=False), 
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),        
            )

	#Residual Block 2
        self.r2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False), 
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
            )

	#Maxpool layer with kernel_size=4        
        self.pool = nn.MaxPool2d(4, 4)

	#Fully connected layer
        self.fc = nn.Linear(in_features = 512, out_features = 10, bias=False)

      


    def forward(self, x):

        preplayer = self.preplayer(x) 

        x1 = self.x1(preplayer)
        r1 = self.r1(x1)
        layer1 = x1+r1

        layer2 = self.layer2(layer1)

        x3 = self.x2(layer2)
        r2 = self.r2(x3)
        layer3 = r2+x3

        maxpool = self.pool(layer3)

        #flattening the tensor
        x = maxpool.view(maxpool.size(0),-1)
        fc = self.fc(x)

        return F.log_softmax(fc.view(-1,10), dim=-1)
