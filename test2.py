from torch import nn
from torch.utils.data import dataset
from torch.utils.data import  dataloader
import torchvision.datasets as dset
import torch.optim
import torchvision.transforms as transforms
from tqdm.auto import tqdm
import numpy as np
import json
from model import  AlexNet,MaskedConv2d
import  util
from model import  Alexnet_pruned

train_loader,validation_loader,test_loader = util.load_transformd_CIFAR10_data()

device = "cuda" if torch.cuda.is_available() else "cpu"
model=Alexnet_pruned()
path='./lr0.001_adam_l2_1e-5/net_param.pkl'


model.load_state_dict(torch.load(path))

model.to(device)
criterion = nn.CrossEntropyLoss()
test_accs = []
test_losses = []
for batch in test_loader:
    imgs, labels = batch

    with torch.no_grad():
        logits = model(imgs.to(device))

    loss = criterion(logits, labels.to(device))

    acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

    test_accs.append(acc)
    test_losses.append(loss)


test_acc = sum(test_accs) / len(test_accs)
test_loss = sum(test_losses) / len(test_losses)
print("final testing " + "   test loss " + str(test_loss) + "   test acc " + str(test_acc))



