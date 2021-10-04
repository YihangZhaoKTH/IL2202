from torch import nn
from torch.utils.data import dataset
from torch.utils.data import  dataloader
import torchvision.datasets as dset
import torch.optim
import torch.nn.functional as F

class AlexNet(nn.Module):
    '''
    original Alexnet without compression
    '''
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4, 2),#0
            nn.BatchNorm2d(96),#1
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 192, 5, 1, 2),#4
            nn.BatchNorm2d(192),#5
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(192, 384, 3, 1, 1),#8
            nn.BatchNorm2d(384),#9
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),#11
            nn.BatchNorm2d(256),#12
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, 1, 1),#14
            nn.BatchNorm2d(128),#15
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )
        self.classifier = nn.Sequential(

            nn.Linear(128 * 6 * 6, 4096),#0
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),#3
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 10),#6
        )

    def forward(self, x):
        x = self.features(x)
        x=x.flatten(1)
        x = self.classifier(x)
        return x


class MaskedConv2d(nn.Conv2d):# there are out_channels filters
    def  __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,bias=True,groups=1):
        super(MaskedConv2d, self).__init__(in_channels,out_channels,kernel_size,stride,padding,bias=bias,groups=groups )
        self.weight_mask=torch.ones_like(self.weight).cuda()
        self.bias_mask=torch.ones_like(self.bias).cuda()
        self.groups=groups
        self.mask_flag=False
        self.count=0

        #self.prev_conv_layer_mask=False


    def set_mask(self , number):
        self.weight_mask.data[number,:,:,:]=0
        self.bias_mask.data[number]=0
        self.count+=1
        self.mask_flag=True
    def clear_mask(self,number):
        self.weight_mask.data[number,:,:,:]=1
        self.bias_mask.data[number]=1
        self.count-=1
        if self.count ==0:
            self.mask_flag = False
    def clear_all_mask(self):
        self.weight_mask.data[:,:,:,:]=1
        self.bias_mask.data[:]=1
        self.count=0
        self.mask_flag=False
    def forward(self, x):
        my_weight=self.weight*self.weight_mask
        my_bias=self.bias*self.bias_mask

        return F.conv2d(x,my_weight,my_bias,self.stride,self.padding,groups=self.groups)



class MaskedBatchNorm2d(nn.BatchNorm2d):
    def __init__(self,num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
        super(MaskedBatchNorm2d, self).__init__(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None )
        self.weight_mask=torch.ones_like(self.weight)
        self.bias_mask=torch.ones_like(self.bias)

    def set_mask(self,number):
        self.weight_mask[number]=0
        self.bias_mask[number]=0
    def clear_mask(self,number):
        self.weight_mask[number]=1
        self.bias_mask[number]=1





class Alexnet_pruned(nn.Module):
    def __init__(self):
        super(Alexnet_pruned, self).__init__()
        self.features = nn.Sequential(
            MaskedConv2d(3, 96, 11, 4, 2),  # 0
            nn.BatchNorm2d(96),  # 1
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            MaskedConv2d(96, 192, 5, 1, 2),  # 4
            nn.BatchNorm2d(192),  # 5
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            MaskedConv2d(192, 384, 3, 1, 1),  # 8
            nn.BatchNorm2d(384),  # 9
            nn.ReLU(),
            MaskedConv2d(384, 256, 3, 1, 1),  # 11
            nn.BatchNorm2d(256),  # 12
            nn.ReLU(),
            MaskedConv2d(256, 128, 3, 1, 1),  # 14
            nn.BatchNorm2d(128),  # 15
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )
        self.classifier = nn.Sequential(

            nn.Linear(128 * 6 * 6, 4096),  # 0
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),  # 3
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 10),  # 6
        )
    #def sort0(self):
    #    tmp=[]
    #    for i in range(96):
    #        tmp.append( torch.sum(  torch.abs(self.features[i].data)))
    #    tmp.sort()



    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x






class Alexnet_CONV_Compressed(nn.Module):
    '''
    only the conv layers are compressed
    '''
    def __init__(self):
        super().__init__()
        self.channel=[3,96,192,384,256,128]
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4, 2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),

            #nn.Conv2d(96, 192, 5, 1, 2),
            #nn.BatchNorm2d(192),
            #nn.ReLU(),
            #nn.MaxPool2d(3, 2),
            nn.Sequential(
                nn.Conv2d(96,96,5,1,2,groups=96),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                nn.Conv2d(96,192,1),
                nn.MaxPool2d(3, 2),
            ),


            #nn.Conv2d(192, 384, 3, 1, 1),
            #nn.BatchNorm2d(384),
            #nn.ReLU()

            nn.Sequential(
                nn.Conv2d(192,192,3,1,1,groups=192),
                nn.BatchNorm2d(192),
                nn.ReLU(),
                nn.Conv2d(192,384,1),
            ),

            #nn.Conv2d(384, 256, 3, 1, 1),
            #nn.BatchNorm2d(256),
            #nn.ReLU(),
            nn.Sequential(
                nn.Conv2d(384,384,3,1,1,groups=384),
                nn.BatchNorm2d(384),
                nn.ReLU(),
                nn.Conv2d(384,256,1),

            ),



            #nn.Conv2d(256, 128, 3, 1, 1),
            #nn.BatchNorm2d(128),
            #nn.ReLU(),

            nn.Sequential(
                nn.Conv2d(256,256,3,1,1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256,128,1),
            ),

            nn.MaxPool2d(3, 2),
        )
        self.classifier = nn.Sequential(

            nn.Linear(128 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

class Alexnet_FCLayer_Compressed(nn.Module):
    def __init__(self):
        super().__init__()
        self.K=1
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4, 2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 192, 5, 1, 2),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(192, 384, 3, 1, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )
        self.classifier = nn.Sequential(

            nn.Linear(128 * 6 * 6, self.K),
            nn.ReLU(),
            nn.Linear(self.K,4096),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(4096,self.K),
            nn.ReLU(),
            nn.Linear(self.K,4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
