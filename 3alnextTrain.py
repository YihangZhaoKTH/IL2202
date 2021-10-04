
from torch import nn
from torch.utils.data import dataset
from torch.utils.data import  dataloader
import torchvision.datasets as dset
import torch.optim
import torchvision.transforms as transforms
from tqdm.auto import tqdm
import numpy as np
import json

from model import Alexnet_CONV_Compressed
'''
only compress the conv layer
'''
#when training, set  testing =0
testing=1

'''
class network(nn.Module):
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
            nn.Dropout(),
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
'''
N=224
Batch=128
n_epochs = 20
learning_rate=0.01
transform=transforms.Compose([
    transforms.Resize((N,N)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
])





'''
train_dataset = dset.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = dset.CIFAR10(root='./data', train=False, transform=transform, download=True)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Batch, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=Batch, shuffle=True)
'''
##############

train_data = dset.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_data = dset.CIFAR10(root='./data', train=False, transform=transform, download=True)
#train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=Batch, shuffle=True)



train_len = len(train_data)
test_len = len(test_data)

indices = range(len(train_data))
indices_train = indices[:40000]
indices_val = indices[40000:]

sampler_train = torch.utils.data.sampler.SubsetRandomSampler(indices_train)
sampler_val = torch.utils.data.sampler.SubsetRandomSampler(indices_val)

train_loader = torch.utils.data.DataLoader(dataset =train_data,
                                                batch_size = Batch,
                                                sampler = sampler_train
                                               )
validation_loader = torch.utils.data.DataLoader(dataset=train_data,
                                          batch_size=Batch,
                                          sampler = sampler_val
                                         )
#########

#########train###########
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Alexnet_CONV_Compressed().to(device)
model.device = device
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-5)

list_train_acc=[]
list_val_acc=[]

for epoch in range(n_epochs):
    # These are used to record information in training.
    train_losses = []
    train_accs = []
    model.train()
    for batch in train_loader:

        imgs,labels=batch
        logits = model(imgs.to(device))
        loss = criterion(logits, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        train_losses.append(loss.item())
        train_accs.append(acc)
        if testing==1:
            break



    train_acc=sum(train_accs)/len(train_accs)

    list_train_acc=np.append(list_train_acc,train_acc.numpy())

    train_loss=sum(train_losses)/len(train_losses)
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
    model.eval()
    val_accs=[]
    val_losses=[]
    for batch in validation_loader:
        imgs,labels=batch

        with torch.no_grad():
            logits=model(imgs)

        loss=criterion(logits,labels)

        acc=(logits.argmax(dim=-1) == labels.to(device)).float().mean()

        val_accs.append(acc)
        val_losses.append(loss)
        if testing ==1:
            break



    val_acc=sum(val_accs)/len(val_accs)
    list_val_acc=np.append(list_val_acc,val_acc.numpy())

    val_loss=sum(val_losses)/len(val_losses)
    print(f"[ valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {val_loss:.5f}, acc = {val_acc:.5f}")


##################testing############
test_accs = []
test_losses = []
model.eval()
for batch in test_loader:
    imgs, labels = batch

    with torch.no_grad():
        logits = model(imgs)

    loss = criterion(logits, labels)

    acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

    test_accs.append(acc)
    test_losses.append(loss)
    if testing == 1:
        break



test_acc = sum(test_accs) / len(test_accs)
test_loss = sum(test_losses) / len(test_losses)
print("final testing " + str(epoch) + "   test loss " + str(test_loss) + "   test acc " + str(test_acc))

torch.save(model.state_dict(), 'model_param/net_param3.pkl')


dic={}
dic["train_acc"]=list_train_acc.tolist()
dic["val_acc"]=list_val_acc.tolist()
with open("./3/acc.json","w") as f:
    json.dump(dic,f)



'''
model2=network()
model2.load_state_dict(torch.load('./model_param/net_param3.pkl'))
model2.eval()
'''











