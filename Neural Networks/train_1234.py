# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import division, print_function
# import torch.multiprocessing as mp
# mp.set_start_method('spawn')
import copy
import os
import time
import zarr
import bsddb3

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skm
import torch
import torch.nn as nn
import torch.optim as optim
# import torchvision
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import model
import h5py

import dataset2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Get a batch of training data
# inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
# out = torchvision.utils.make_grid(inputs)


def train_model(workdir,
                modeldir,
                model,
                criterion,
                optimizer,
                scheduler,
                dataloaders,
                num_epochs=224):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    model_dir = os.path.join(workdir, modeldir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    for epoch in range(num_epochs):
        print(time.asctime())
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            temp_count = 0
            for inputs, labels in dataloaders[phase]:
                temp_count += 1
                print(temp_count)
                if temp_count % 500 == 0:
                    print(temp_count)
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # save models
            print('save model for epoch #{}'.format(epoch))
            model_path = os.path.join(model_dir, 'epoch_{}.pth'.format(epoch))
            torch.save(model.state_dict(), model_path)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss,
                                                       epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(str(preds[j])))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def getValConfusMatrix(model, val_dataloader):
    # load the state_dict
    model.load_state_dict(torch.load('./epoch_24.pth'))
    model.eval()

    re_label = []
    re_pred = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            print(i)
            inputs = inputs.to(device)
            labels = labels.to(device)
            re_label.extend(labels.view(-1).tolist())

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            print(outputs, preds)
            re_pred.extend(preds.view(-1).tolist())
    print(len(re_pred), len(re_label))
    coffusion_matrix = skm.confusion_matrix(np.array(re_label),
                                            np.array(re_pred))
    return coffusion_matrix


if __name__ == '__main__':
    zarr_file_list = ['./dataset/train_{}.lmdb'.format(i+1) for i in range(4)]
    # zarr_file_list = [zarr.LMDBStore(i) for i in h5_path_list]
    # h5f = h5py.File(h5_path, 'r', swmr=True)
    datasets = {x: dataset2.MyDataset(zarr_file_list, is_trainning=(x=='train'))
                for x in ['train', 'val']}
    dataloaders = {
        x: torch.utils.data.DataLoader(datasets[x],
                                       batch_size=1000,
                                       shuffle=True,
                                       num_workers=4,
                                       pin_memory=True)
        for x in ['train', 'val']
    }

    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    # class_names = image_datasets['train'].classes
    # model_ft = models.resnet101(pretrained=True)
    model_ft = model.Net(204, 19)
    # num_ftrs = model_ft.fc.in_features
    # model_ft.fc = nn.Linear(num_ftrs, 8)
    print(device)
    model_ft = model_ft.to(device)
    criterion = nn.NLLLoss()
    model = model_ft
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft,
                                        step_size=7,
                                        gamma=0.1)
    # print(model.state_dict().keys())
    # out = getValConfusMatrix(model_ft, dataloaders['val'])
    # print(out)
    # print("Model's state_dict:")
    # for param_tensor in model_ft.state_dict():
    #     print(param_tensor, "\t", model_ft.state_dict()[param_tensor].size())

    # print("Optimizer's state_dict:")
    # for var_name in optimizer_ft.state_dict():
    #     print(var_name, "\t", optimizer_ft.state_dict()[var_name])
    model_ft = train_model('./',
                           'checkpoint_{}'.format(1234),
                           model_ft,
                           criterion,
                           optimizer_ft,
                           exp_lr_scheduler,
                           dataloaders,
                           num_epochs=25)
    # torch.save(model.state_dict(), './checkpoint/best.pth')
