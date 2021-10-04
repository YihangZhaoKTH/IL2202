from torch import nn
from torch.utils.data import dataset
from torch.utils.data import  dataloader
import torchvision.datasets as dset
import torch.optim
import torchvision.transforms as transforms
from tqdm.auto import tqdm
import numpy as np
import json
from model import  AlexNet,MaskedConv2d,Alexnet_pruned
import  util


TESTING_MODE=1

N=224
Batch=128
n_epochs = 20
learning_rate=0.0001
transform=transforms.Compose([
    transforms.Resize((N,N)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(90),
    transforms.RandomGrayscale(0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
])
transform2=transforms.Compose([
    transforms.Resize((N,N)),
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomAffine(90),
    #transforms.RandomGrayscale(0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
])






##############

train_data = dset.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_data = dset.CIFAR10(root='./data', train=False, transform=transform2, download=True)
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

device = "cuda" if torch.cuda.is_available() else "cpu"
#device="cpu"
model=Alexnet_pruned()

model.device=device
model.to(device)

path="./1/60_checkpoint"


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

epoch,model_param,optimizer_param,loss=util.load_check_point(path)

model.load_state_dict(model_param)
optimizer.load_state_dict(optimizer_param)
param_dic=model.state_dict()
for param in param_dic:
    print(param, param_dic[param].size())
#print(param_dic["features.0.weight"].data[1,:,:,:].shape)

model.features[14].set_mask(10)




test_accs = []
test_losses = []
model.eval()
for batch in test_loader:
    imgs, labels = batch

    with torch.no_grad():
        logits = model(imgs.to(device))

    loss = criterion(logits, labels.to(device))

    acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

    test_accs.append(acc)
    test_losses.append(loss)
    if TESTING_MODE == 1:
        break
test_acc = sum(test_accs) / len(test_accs)
test_loss = sum(test_losses) / len(test_losses)
print("final testing " + str(epoch) + "   test loss " + str(test_loss) + "   test acc " + str(test_acc))




