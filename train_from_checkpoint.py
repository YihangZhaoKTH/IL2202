from torch import nn
from torch.utils.data import dataset
from torch.utils.data import  dataloader
import torchvision.datasets as dset
import torch.optim
import torchvision.transforms as transforms
from tqdm.auto import tqdm
import numpy as np
import json

from model import  AlexNet
import util
'''
fine tune the fully connected layer firstly
'''


learning_rate=0.000001
TESTING_MODE=0
n_epochs=150#150

train_loader,validation_loader,test_loader=util.load_transformd_CIFAR10_data()



device = "cuda" if torch.cuda.is_available() else "cpu"
model=AlexNet()
path='./lr0.001_adam_l2_1e-5/90_checkpoint'
epoch,model_state_dict,optimizer_state_dict,loss=  util.load_check_point(path)

model.load_state_dict(model_state_dict)

model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
optimizer.load_state_dict(optimizer_state_dict)




list_train_acc=[]
list_val_acc=[]
while epoch <n_epochs:
    print('epoch '+str(epoch+1))
    # These are used to record information in training.
    train_losses = []
    train_accs = []
    model.train()
    for batch in train_loader:

        imgs,labels=batch
        #print(imgs.shape)
        #print(imgs[0])
        logits = model(imgs.to(device))
        loss = criterion(logits, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        train_losses.append(loss.item())
        train_accs.append(acc)
        if TESTING_MODE==1:
            break



    train_acc=sum(train_accs)/len(train_accs)

    list_train_acc=np.append(list_train_acc,train_acc.cpu().numpy())

    train_loss=sum(train_losses)/len(train_losses)
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
    model.eval()
    val_accs=[]
    val_losses=[]
    for batch in validation_loader:
        imgs,labels=batch

        with torch.no_grad():
            logits=model(imgs.to(device))

        loss=criterion(logits,labels.to(device))

        acc=(logits.argmax(dim=-1) == labels.to(device)).float().mean()

        val_accs.append(acc)
        val_losses.append(loss)
        if TESTING_MODE==1:
            break




    val_acc=sum(val_accs)/len(val_accs)
    list_val_acc=np.append(list_val_acc,val_acc.cpu().numpy())

    val_loss=sum(val_losses)/len(val_losses)
    print(f"[ valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {val_loss:.5f}, acc = {val_acc:.5f}")
    if epoch %10==0 and ( epoch != 0 ):
        util.save_check_point(epoch,model,optimizer,train_loss)
    epoch=epoch+1


#util.train(model,optimizer,criterion,device,n_epoch,train_loader,validation_loader,test_loader)
##################testing############
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

torch.save(model.state_dict(), 'model_param/net_param.pkl')


dic={}
dic["train_acc"]=list_train_acc.tolist()
dic["val_acc"]=list_val_acc.tolist()
with open("./1/final_acc.json","w") as f:
    json.dump(dic,f)



