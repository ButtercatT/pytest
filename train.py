import time

import numpy as np
import torch
import torch.nn as nn
from model_Resnet import resnet34
from torchvision import transforms
from HW3 import read_file, ImgDataset
import matplotlib.pyplot as plt
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# parameters
num_epoch = 35
num_classes = 11
batch_size = 32
learning_rate = 0.003
t_loss = []
v_loss = []

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

# read the file
pname = "./"
training_x, training_y = read_file(os.path.join(pname, "training"), True)
print("Size of training data = {}".format(len(training_x)))

val_x, val_y = read_file(os.path.join(pname, "validation"), True)
print("Size of validation data = {}".format(len(val_x)))

# dataloader
train_dataset = torch.utils.data.DataLoader(datast=ImgDataset(training_x, training_y, transform=data_transform['train'])
                                            , batch_size=batch_size
                                            , shuffle=True
                                            )
val_dataset = torch.utils.data.DataLoader(dataset=ImgDataset(val_x, val_y, transform=data_transform['val'])
                                          , batch_size=batch_size
                                          , shuffle=False
                                          )
# model
model = resnet34(num_classes=num_classes).to(device)

# loss
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# record model weights
save_path = './resNet34.pth'

# train
for epoch in range(num_epoch):
    model.train()
    train_acc = 0.0
    val_acc = 0.0
    epoch_start_time = time.time()
    for i, data in enumerate(train_dataset):
        # forward pass
        y_head = model(data[0].to(device))
        loss = loss_function(y_head, data[1].to(device))

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record the acc:
        train_acc += np.sum(np.argmax(y_head.cuda().data.cpu().numpy(), axis=1) == data[1].numpy())
        t_loss.append(1 - train_acc / len(training_x))
    # validation
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_dataset):
            y_head = model(data[0].to(device))
            val_acc += np.sum(np.argmax(y_head.cuda().data.cpu().numpy(), axis=1) == data[1].numpy())

        v_loss.append(1 - val_acc / len(val_x))
        print('timecost:{} processing:[{}/{}] training acc:{}    val_acc:{}  '.format(
            time.time() - epoch_start_time,
            epoch + 1,
            num_epoch,
            train_acc / len(training_x),
            val_acc / len(val_x))
        )
# plot the result
plt.plot(t_loss, color='red', label='Training error')
plt.plot(v_loss, color='green', label='Testing error')
plt.legend()
plt.xlabel('iteration times')
plt.ylabel('error')
plt.show()
