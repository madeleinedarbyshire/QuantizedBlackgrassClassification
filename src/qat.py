import argparse
import copy
import sys
import time
import torch
import torch.nn as nn

from model import add_channels
from quantizable_resnet import resnet18
from torch.optim import lr_scheduler, SGD
from utils import load_data

class QuantizedResNet18(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedResNet18, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.model_fp32 = model_fp32

    def forward(self, x):
        x = self.quant(x)
        x = self.model_fp32(x)
        x = self.dequant(x)
        return x

def train_one_epoch(model, criterion, optimizer, scheduler, dataloader, device):
    model.train()

    running_loss = 0.0
    running_corrects = 0
    total = 0

    for inputs, labels in dataloader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total += inputs.size(0)

    scheduler.step()
    
    epoch_loss = running_loss / total
    epoch_acc = running_corrects.double() / total

    print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Total: {total:.0f}')

def validation(model, criterion, optimizer, dataloader, device):
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for inputs, labels in dataloader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total += inputs.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = running_corrects.double() / total

    print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Total: {total:.0f}')

    return epoch_acc

def main(model_path, channels, img_root, resolution, lr, momentum):
    model_name, file_suffix = model_path.split('/')[-1].split('.')
    qat_model_filepath = f'models/iros/qat/{model_name}.{file_suffix}'
    txt_filename = f'out/{model_name}_qat.txt'
    # dataloaders = load_data(resolution, channels, ['train', 'test'], 'resources/qat_dataset.csv', path=img_root, batch_size=64)
    qat_dataloader = load_data(resolution, channels, ['train'], 'resources/dataset_qat_2001.csv', path=img_root, batch_size=64, pin_memory=False)
    dataloaders = load_data(resolution, channels, ['val', 'test'], 'resources/dataset_2001.csv', path=img_root, batch_size=64, pin_memory=False)
    cal_dataloader = load_data(resolution, channels, ['cal'], 'resources/dataset_calibration_2001.csv', path=img_root, batch_size=64, pin_memory=False)
    model = resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
    model = add_channels(model, channels)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    # model.eval()
    # model = fuse_modules(model)

    input_fp32, _ = next(iter(cal_dataloader['cal']))

    quantized_model = QuantizedResNet18(model_fp32=model)
    quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    print('Preparing QAT...')
    torch.quantization.prepare_qat(quantized_model, inplace=True)
    quantized_model(input_fp32)


    optimizer = SGD(quantized_model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=6, eta_min=0.0001)

    best_acc = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    dataloaders.update(qat_dataloader)
    with open(txt_filename, 'w') as sys.stdout:
        for nepoch in range(6):
            print(f'Epoch {nepoch}/5')
            print('-' * 10)

            train_one_epoch(quantized_model, criterion, optimizer, scheduler, dataloaders['train'], torch.device('cpu'))

            print('Validating...')
            trained = torch.quantization.convert(quantized_model, inplace=False)
            trained.eval()
            acc = validation(trained, criterion, optimizer, dataloaders['val'], torch.device('cpu'))
            if acc > best_acc:
                best_acc = acc
                print(f' Best Acc: {best_acc:.4f}')
                print('Testing...')
                validation(trained, criterion, optimizer, dataloaders['test'], torch.device('cpu'))
                best_model_wts = copy.deepcopy(trained.state_dict())
                torch.save(best_model_wts, qat_model_filepath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--channels', nargs='+', type=str, default=['red', 'green', 'blue', 'nir', 'red_edge'])
    parser.add_argument('--img-root', type=str, default='../data')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    args = parser.parse_args()
    main(args.model_path, args.channels, args.img_root, args.resolution, args.lr, args.momentum)