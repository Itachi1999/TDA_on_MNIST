'''This model is similar to ResNet34 Model'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
from torchsummary import summary
# Paths

# train_path = './FNAC/train/'
# validation_path='./FNAC/validation/'

# # Data Loading

# # TRANSFORM
# transform =transforms.Compose([
#        transforms.RandomCrop((224,224)),
#        transforms.RandomHorizontalFlip(),
#        transforms.RandomVerticalFlip(),         
#        transforms.RandomRotation(30),
#        transforms.ToTensor(),
# 		   transforms.Normalize(mean = [ 0.5, 0.5, 0.5 ],
# 				                    std =  [ 0.5, 0.5, 0.5 ]),
#                           ])
                        
# # Loading data from a image folder

# train_data = datasets.ImageFolder(train_path ,transform)
# validation_data = datasets.ImageFolder(validation_path , transform)

# # Loading data in batches                                       
                                       
# train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=2)

# validation_loader = DataLoader(validation_data , batch_size=1, shuffle=True ,num_workers=2)

# classes = ('B','M')

# net= models.resnet18()
# net.fc=nn.Linear(512,2)
# net = net.cuda() 

# # Define the loss Function

# criterion = nn.NLLLoss(size_average = True)
# optimizer = optim.Adam(net.parameters(),lr=0.0001)

# train_loss_vs_epoch=[]
# validation_loss_vs_epoch=[]

  
# # Training the network 
# min_validation = 9999
# for epoch in range(200) :  
#     train_loss = 0.0
    
 
#     for i,batch in enumerate(train_loader,0):
#         net=net.train(True) 
     
#         inputs, labels = batch
#         inputs, labels = inputs.cuda(), labels.cuda()
#         inputs, labels = Variable(inputs), Variable(labels) 
#         outputs = net(inputs)
#         loss = criterion(F.log_softmax(outputs), labels)
       
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()
#     train_loss_vs_epoch.append(train_loss/len(train_loader))

   

    
#     validation_loss = 0.0
#     for batch in validation_loader :
#         net = net.train(False)
#         inputs, labels = batch
#         inputs, labels = inputs.cuda(),labels.cuda();
#         inputs, labels = Variable(inputs), Variable(labels) 
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)        
#         validation_loss += loss.item()
     
#     validation_loss_vs_epoch.append(validation_loss/len(validation_loader))  
     
     
#     plt.plot(train_loss_vs_epoch,'r',validation_loss_vs_epoch,'b')
#     plt.savefig('plot_resnet18_loss.png')
    
#     if validation_loss < min_validation :
#         min_validation = validation_loss
#         torch.save(net,'./model/res18_best.pt');
#         print('model saved')
  
#     print('[epoch: %d] train loss: %.6f, val loss= %0.6f' %(epoch+1, train_loss/len(train_loader), validation_loss/len(validation_loader)))
        
# print('Finished Training')


class conv_res_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_res_block,self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True)
        self.batch1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True)
        self.batch2 = nn.BatchNorm2d(ch_out)


    def forward(self,x):
        x1 = self.conv1(x)
        x1 = self.batch1(x1)
        x1 = F.relu(x1, inplace = True)
        x1 = self.conv2(x1)
        x1 = self.batch2(x)
        x = x + x1
        x = F.relu(x, inplace = True)
        return x



class Model(torch.nn.Module):
    def __init__(self, ip_channel = 3, op_channel = 1) -> None:
        super(Model, self).__init__()

        self.convBlock1 = conv_res_block(ip_channel, 64)
        self.convBlock2 = conv_res_block(64, 128)
        self.convBlock3 = conv_res_block(128, 256)


    def forward(self, x):
        x = self.convBlock1(x)


