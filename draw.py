from torch import nn
from torch.utils.data import dataset
from torch.utils.data import  dataloader
import torchvision.datasets as dset
import torch.optim
import torchvision.transforms as transforms
import numpy as np
import json
import matplotlib.pyplot  as plt
from matplotlib.pyplot import MultipleLocator

FUNCTION = "sensitive"
if FUNCTION == "sensitive":
    with open("1/sensitive_L2.json") as f:
        result = json.load(f)
    x = ([0,10,20,30,40,50,60,70,80,90])
    leg = ["conv-0"," conv-4"," conv-8"," conv-11"," conv-14"]
    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    for i in range(5):
        y= result["1"][i]

        ax.plot(x,y,label = leg[i])  # Plot some data on the axes.


    ax.set_xlabel("pruned percentage(%)")
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy after pruning ")

    ax.legend()
    ax.grid()

    fig.savefig("./1/sensitive_L2.png")
    plt.show()
else:
    with open("lr0.001_adam_l2_1e-5/final_acc.json") as f:
        dic = json.load(f)
    train_acc = dic["train_acc"]
    val_acc = dic["val_acc"]
    print(len(val_acc))
    x=[ i+1 for i in range(len(val_acc))]

    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.plot(x,train_acc,label=" training set")
    ax.plot(x,val_acc,label = "validation set")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy ")
    ax.set_title("Accuracy of Training set and Validation set")
    ax.legend()
    fig.savefig("./1/train.png")
    plt.show()


