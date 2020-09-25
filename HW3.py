import time
from typing import Any

import torch
import cv2
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset

is_using_gpu: bool = True
"""
set to use GPU or not.
True: use GPU
"""


def read_file(path: str, flag: bool, size=224):
    """
    used on type like 'class_no' files
    :param path: picture's path
    :param flag: is it a testing data file or not
    :return: get the data matrix (and it's type belonging)

    """
    img_path = sorted(os.listdir(path))
    img_num = len(img_path)
    img_data = np.zeros((img_num, 224, 224, 3), dtype=np.uint8)
    img_label = np.zeros(img_num, dtype=np.uint8)
    for lps, no in enumerate(img_path):
        img = cv2.imread(os.path.join(path, no))
        img_data[lps, :, :] = cv2.resize(img, (224, 224))
        if flag:
            img_label[lps] = int(no.split("_")[0])
    if flag:
        return img_data, img_label
    else:
        return img_data


# Device configuration
device = torch.device('cuda' if is_using_gpu else 'cpu')
# parameters
num_iters = 31
num_classes = 10
batch_size = 64
learning_rate = 0.001
t_loss = []
v_loss = []
# pre dealing
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),  # 翻转
    transforms.RandomRotation(15),  # 随机旋转图片
    transforms.ToTensor(),  # Tensor，normalize 到 [0,1] (data normalization)
])
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])


# construct a dataset by overriding
class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        return X


# CNN model
class CovNet(nn.Module):

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self):
        super(CovNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),  # normalization (for 64 features)
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0)  # 64*64*64
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0)  # 128*32*32
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0)  # 256*16*16
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0)  # 512*8*8
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0)  # 512*4*4
        )

        # then doing the full-connecting-forwardNN
        self.fc = (nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 11)
        ))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)  # Flatten
        out = self.fc(out)
        return out


# read the file
pname = "./"
training_x, training_y = read_file(os.path.join(pname, "training"), True)
print("Size of training data = {}".format(len(training_x)))

val_x, val_y = read_file(os.path.join(pname, "validation"), True)
print("Size of validation data = {}".format(len(val_x)))

testing_x = read_file(os.path.join(pname, "testing"), False)
print("Size of testing data = {}".format(len(testing_x)))

# # Data loader
# train_loader = torch.utils.data.DataLoader(dataset=ImgDataset(training_x, training_y, train_transform)
#                                            , batch_size=batch_size
#                                            , shuffle=True)
#
test_loader = torch.utils.data.DataLoader(dataset=ImgDataset(testing_x, transform=test_transform)
                                          , shuffle=False
                                          , batch_size=batch_size)

# validation_loader = torch.utils.data.DataLoader(dataset=ImgDataset(val_x, val_y, test_transform)
#                                                 , batch_size=batch_size
#                                                 , shuffle=False)
# initial the model
# model = CovNet().to(device)
#
# # Loss function,optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# # training
# for lps in range(num_iters):
#     train_acc = 0.0
#     train_loss = 0.0
#     val_acc = 0.0
#     val_loss = 0.0
#     epoch_start_time = time.time()
#     for i, data in enumerate(train_loader):
#         model.train()  # if using dropout or BatchNorm2d,this line is recommended to run
#         # forward pass
#         output = model(data[0].to(device))  # probabilities of the img in different class
#         loss = criterion(output, data[1].to(device))
#
#         # backward pass
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         train_acc += np.sum(np.argmax(output.cuda().data.cpu().numpy(), axis=1) == data[1].numpy())
#         train_loss += loss.item()
#
#     # Test the model by using the validation_loader
#     # IMPORTANT:
#     model.eval()
#     with torch.no_grad():
#
#         for j, data in enumerate(validation_loader):
#             y_head = model(data[0].to(device))
#             loss = criterion(y_head, data[1].to(device))
#             val_acc += np.sum(np.argmax(y_head.cuda().data.cpu().numpy(), axis=1) == data[1].numpy())
#             val_loss += loss.item()
#
#         t_loss.append(1 - train_acc / len(training_x))
#         v_loss.append(1 - val_acc / len(val_x))
#         print('timecost:{} processing:[{}/{}] training acc:{}  training loss:{}  val_acc:{}  val_loss:{}'.format(
#             time.time() - epoch_start_time,
#             lps + 1,
#             num_iters,
#             train_acc / len(training_x),
#             train_loss / len(training_x),
#             val_acc / len(val_x),
#             val_loss / len(val_x)
#         )
#         )

# # plot the result
# plt.plot(t_loss, color='red', label='Training accuracy')
# plt.plot(v_loss, color='green', label='Testing accuracy')
# plt.legend()
# plt.xlabel('iteration times')
# plt.ylabel('error')
# plt.show()

# TODO 训练出较好的模型，将所有数据集合进行最后的训练
combine_x = np.concatenate((training_x, val_x), axis=0)
combine_y = np.concatenate((training_y, val_y), axis=0)
conbine_loader = torch.utils.data.DataLoader(dataset=ImgDataset(combine_x, combine_y, train_transform)
                                             , shuffle=True
                                             , batch_size=batch_size)
# TODO train the best model
best_model = CovNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(best_model.parameters(), lr=learning_rate)
# training
for lps in range(num_iters):
    train_acc = 0.0
    train_loss = 0.0
    epoch_start_time = time.time()
    for i, data in enumerate(conbine_loader):
        best_model.train()  # if using dropout or BatchNorm2d,this line is recommended to run
        # forward pass
        output = best_model(data[0].to(device))  # probabilities of the img in different class
        loss = criterion(output, data[1].to(device))

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc += np.sum(np.argmax(output.cuda().data.cpu().numpy(), axis=1) == data[1].numpy())
        train_loss += loss.item()
    print('timecost:{} processing:[{}/{}] training acc:{}  training loss:{}'.format(
        time.time() - epoch_start_time,
        lps + 1,
        num_iters,
        train_acc / len(combine_x),
        train_loss / len(combine_x),
    )
    )
# run testing_data
best_model.eval()
y_pred = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        res = best_model(data.to(device))
        y_head = np.argmax(data.cuda().data.cpu().numpy(), axis=1)
        for y in y_head:
            y_pred.append(y)

with open("predict.csv", 'w') as f:
    f.write('Id,Category\n')
    for i, y in enumerate(y_pred):
        f.write('{},{}\n'.format(i, y))
