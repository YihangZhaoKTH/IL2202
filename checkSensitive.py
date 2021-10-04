from torch import nn
from torch.utils.data import dataset
from torch.utils.data import  dataloader
import torchvision.datasets as dset
import torch.optim
import torchvision.transforms as transforms
import numpy as np
import json
import torch.nn.functional as F
import util
from model import AlexNet, Alexnet_pruned

train_loader,validation_loader,test_loader = util.load_transformd_CIFAR10_data()

device = "cuda" if torch.cuda.is_available() else "cpu"
model=Alexnet_pruned()
path='./lr0.001_adam_l2_1e-5/net_param.pkl'

TESTING_MODE=1
model.load_state_dict(torch.load(path))

model.to(device)
criterion = nn.CrossEntropyLoss()

test_accs = []
test_losses = []
model.eval()

conv_list=[ (0,0)  for i in range(15)]
conv_list[0]=(96,3*11*11)
conv_list[4]=(192,5*5*96)
conv_list[8]=(384,192*3*3)
conv_list[11]=(256,384*3*3)
conv_list[14]=(128,256*3*3)

def check(model,conv_list):
    model.eval()
    list=[0,4,8,11,14]
    result = [[],[],[],[],[]]
    for i in range(5):
        #sub_result=[[],[],[],[],[]]
        a = model.features[list[i]].weight
        a = torch.reshape(a, conv_list[list[i]])

        #a = torch.abs(a)
        a = a*a
        a = torch.sum(a, dim=1, keepdim=True)
        a = torch.argsort(a, dim=0)
        length = conv_list[list[i]][0]
        for j in range(10):
            for k in range( int(length *0.1*j)):
                model.features[list[i]].set_mask(k)



            print(f"conv-layer {list[i] } mask { j } * 10%  ")
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
                if TESTING_MODE == 1:
                    break



            test_acc = sum(test_accs) / len(test_accs)
            test_loss = sum(test_losses) / len(test_losses)
            print("final testing " +"   test loss " + str(test_loss) + "   test acc " + str(test_acc))
            result[i].append(test_acc.cpu().numpy())
            model.features[list[i]].clear_all_mask()

    #result.append(sub_result)
    return result


result=check(model,conv_list)
print(len(result))
print(len(result[0]))

for i in range(5):
    for j in range(10):
        result[i][j]=result[i][j].tolist()

for i in range(5):
    print(result[i])

dic={}
dic["1"]=result

with open("./1/sensitive_L2.json","w") as f:
    json.dump(dic,f)



