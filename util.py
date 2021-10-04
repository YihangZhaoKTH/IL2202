from torch import nn
from torch.utils.data import dataset
from torch.utils.data import  dataloader
import torchvision.datasets as dset
import torch.optim
import torchvision.transforms as transforms
import numpy as np
import json
def save_check_point(epoch,model,optimizer,loss):
    path="./1/"+str(epoch)+"_checkpoint"
    torch.save(
        {
            "epoch":epoch,
            "model_state_dict":model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(),
            "loss":loss
        },
        path
    )
def load_check_point(path):
    checkpoint=torch.load(path)
    return checkpoint["epoch"],checkpoint["model_state_dict"],checkpoint["optimizer_state_dict"],checkpoint["loss"]

def load_transformd_CIFAR10_data():
    N = 224
    Batch = 128
    n_epochs = 50
    learning_rate = 0.0001
    transform = transforms.Compose([
        transforms.Resize((N, N)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(90),
        transforms.RandomGrayscale(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    transform2 = transforms.Compose([
        transforms.Resize((N, N)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomAffine(90),
        # transforms.RandomGrayscale(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    train_data = dset.CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_data = dset.CIFAR10(root='./data', train=False, transform=transform2, download=True)
    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=Batch, shuffle=True)
    train_len = len(train_data)
    test_len = len(test_data)

    indices = range(len(train_data))
    indices_train = indices[:40000]
    indices_val = indices[40000:]

    sampler_train = torch.utils.data.sampler.SubsetRandomSampler(indices_train)
    sampler_val = torch.utils.data.sampler.SubsetRandomSampler(indices_val)

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=Batch,
                                               sampler=sampler_train
                                               )
    validation_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                    batch_size=Batch,
                                                    sampler=sampler_val
                                                    )
    return train_loader,validation_loader,test_loader

def train(model,optimizer,criterion,device,n_epochs,train_loader,validation_loader,test_loader,TESTING_MODE=0):

    list_train_acc = []
    list_val_acc = []

    for epoch in range(n_epochs):
        print('epoch ' + str(epoch + 1))
        # These are used to record information in training.
        train_losses = []
        train_accs = []
        model.train()
        for batch in train_loader:

            imgs, labels = batch
            # print(imgs.shape)
            # print(imgs[0])
            logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            train_losses.append(loss.item())
            train_accs.append(acc)
            if TESTING_MODE == 1:
                break

        train_acc = sum(train_accs) / len(train_accs)

        list_train_acc = np.append(list_train_acc, train_acc.cpu().numpy())

        train_loss = sum(train_losses) / len(train_losses)
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
        model.eval()
        val_accs = []
        val_losses = []
        for batch in validation_loader:
            imgs, labels = batch

            with torch.no_grad():
                logits = model(imgs.to(device))

            loss = criterion(logits, labels.to(device))

            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            val_accs.append(acc)
            val_losses.append(loss)
            if TESTING_MODE == 1:
                break

        val_acc = sum(val_accs) / len(val_accs)
        list_val_acc = np.append(list_val_acc, val_acc.cpu().numpy())

        val_loss = sum(val_losses) / len(val_losses)
        print(f"[ valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {val_loss:.5f}, acc = {val_acc:.5f}")
        if epoch % 10 == 0 and (epoch != 0):
            save_check_point(epoch, model, optimizer, train_loss)

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

    dic = {}
    dic["train_acc"] = list_train_acc.tolist()
    dic["val_acc"] = list_val_acc.tolist()
    with open("./1/final_acc.json", "w") as f:
        json.dump(dic, f)




















