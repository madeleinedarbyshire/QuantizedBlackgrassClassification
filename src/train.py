from __future__ import print_function, division

import argparse
import copy
import sys
import time
import torch
import torch.nn as nn
from torch.optim import lr_scheduler, SGD
import torch.backends.cudnn as cudnn

from model import full_precision_model
from utils import load_data

cudnn.benchmark = True

def powerset(s):
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1, 1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]

def train_model(model, criterion, optimizer, scheduler, dataloaders, device, model_filename, num_epochs=25, save=True):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        epoch_start = time.time() 

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() 
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total = 0

            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

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
                total += inputs.size(0)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / total
            epoch_acc = running_corrects.double() / total

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Total: {total:.0f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                print(f' Best Acc: {best_acc:.4f}')
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, model_filename)

        epoch_duration = time.time() - epoch_start
        print(f'Epoch complete in {epoch_duration // 60:.0f}m {epoch_duration % 60:.0f}s')
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model

def setup_and_train_model(batch_size, channels, img_root, num_epochs, resolution, lr, momentum):
    spectrum_name = ''.join([c[0]+c[-1] for c in channels])
    txt_filename = f'out/2001/resnet18_{num_epochs}_{resolution}_{spectrum_name}_2001.txt'
    model_filename = f'models/2001/resnet18_{num_epochs}_{resolution}_{spectrum_name}_2001.pth'
    with open(txt_filename, 'w') as sys.stdout:
        dataloaders = load_data(resolution, channels, ['train', 'val'], 'resources/dataset_2001.csv', path=img_root, class_path='resources/labels.txt', batch_size=batch_size)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = full_precision_model(channels)
        model = model.to(device)
        optimizer_ft = SGD(model.parameters(), lr=lr, momentum=momentum)
        criterion = nn.CrossEntropyLoss()
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        train_model(model, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, device, model_filename, num_epochs=num_epochs)

def main(batch_size, channels, img_root, num_epochs, resolution, lr, momentum, train_powerset):
    if train_powerset:
        for channels in powerset(['red', 'green', 'blue', 'nir', 'red_edge']):
            print(f'Channels: {channels}')
            setup_and_train_model(batch_size, channels, img_root, num_epochs, resolution, lr, momentum)
            sys.stdout = sys.__stdout__
    else:
        setup_and_train_model(batch_size, channels, img_root, num_epochs, resolution, lr, momentum)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--channels', nargs="+", type=str, default=['red', 'green', 'blue', 'nir', 'red_edge'])
    parser.add_argument('--img-root', type=str, default='../data')
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--train-powerset', type=bool)
    args = parser.parse_args()
    main(args.batch_size, args.channels, args.img_root, args.num_epochs, args.resolution, args.lr, args.momentum, args.train_powerset)