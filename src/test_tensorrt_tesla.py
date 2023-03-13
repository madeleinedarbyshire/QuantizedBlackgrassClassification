import argparse
import copy
import sys
import time
import torch
import torch_tensorrt
import torch.nn as nn
# import torchvision as tv

from calibration_tensorrt import calibrate_model
from model import full_precision_model
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from qat import train_one_epoch, validation
from torch.optim import lr_scheduler, SGD
from torchvision import models, transforms
from utils import load_data

batch_sizes = {512: 128, 256: 1024, 128: 4096}

def parse_precision(precision_str):
    if precision_str == 'float32':
        return torch.float32
    if precision_str == 'float16':
        return torch.float16
    if precision_str == 'int8':
        return torch.int8
    raise ValueError(f'Cannot parse type: {precision_str}')

def add_channels(model, channels):
    weight_indices = {'red': 0, 'green': 1, 'blue': 2, 'nir': 0, 'red_edge': 0}
    weight = model.conv1.weight.clone()
    model.conv1 = nn.Conv2d(len(channels), 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        for i, channel in enumerate(channels):
            model.conv1.weight[:, i] = weight[:, weight_indices[channel]]
    return model

def train_qat(model, dataloaders):
    model.train()
    optimizer = SGD(model.parameters(), lr=0.0001, momentum=0)
    criterion = nn.CrossEntropyLoss()
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=6, eta_min=0.0001)

    best_acc = 0
    best_model = copy.deepcopy(model)
    for nepoch in range(6):
        print(f'Epoch {nepoch}/5')
        print('-' * 10)

        print('Training one epoch...')
        train_one_epoch(model, criterion, optimizer, scheduler, dataloaders['train'], torch.device('cuda'))

        print('Validating...')
        model.eval()
        acc = validation(model, criterion, optimizer, dataloaders['val'], torch.device('cuda'))
        if acc > best_acc:
            best_acc = acc
            print(f' Best Acc: {best_acc:.4f}')
            best_model = copy.deepcopy(model)
    return best_model

def compile_trt_model(model, dataloader, precision, batch_size, num_channels, resolution, debug=False, calibrate=True):
    compile_spec = {'inputs': [torch_tensorrt.Input(min_shape=[1, num_channels, resolution, resolution],
                                                    opt_shape=[batch_size, num_channels, resolution, resolution],
                                                    max_shape=[batch_size, num_channels, resolution, resolution],
                                                    dtype=torch.float32)],
                    'enabled_precisions': precision,
                    'require_full_compilation': True,
                    'truncate_long_and_double': True}



    if precision == torch.int8 and calibrate:
        print('Calibrating with tensor rt...')
        compile_spec['calibrator'] = torch_tensorrt.ptq.DataLoaderCalibrator(dataloader,
                                                                            use_cache=False,
                                                                            algo_type=torch_tensorrt.ptq.CalibrationAlgo.MINMAX_CALIBRATION,
                                                                            device=torch.device('cuda'))

    if debug:
        torch_tensorrt.logging.debug()

    print('Compiling TensorRT module....')
    trt_ts_module = torch_tensorrt.compile(model, **compile_spec)

    return trt_ts_module

def warm_up(model, batch_size, num_channels, resolution):
    for _ in range(5):
        inputs = torch.randn(batch_size, num_channels, resolution, resolution).cuda()
        model(inputs)


def main(channels, img_root, resolution, model_path, precision, test_device, qat):

    print('Channels: ', str(channels), '. Resolution: ', resolution, ' Precision:', precision)

    if precision == torch.int8:
        quant_modules.initialize()

    model_ft = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
    model_ft = add_channels(model_ft, channels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft = model_ft.to(device)
    state_dict = torch.load(model_path)
    model_ft.load_state_dict(state_dict)

    model_ft.eval()

    model_name = model_path.split('/')[-1].split('.')[0]
    if qat:
        filename = f'results/2001/1702/{model_name}_{precision}_tensorrt_qat.csv'
    else:
        filename = f'results/2001/1702/{model_name}_{precision}_tensorrt.csv'
    print('Writing results to file:', filename)
    file = open(filename, 'w')
    file.write(f'Batch_Size, Accuracy, Speed, Total\n')
    file.close()
    
    max_batch_size = batch_sizes[int(resolution)]
    print('Max batch size: ', max_batch_size)
    cal_dataloader = load_data(resolution, channels, ['cal'], 'resources/dataset_calibration_2001.csv', path=img_root, class_path='resources/labels.txt', batch_size=64, test_device=test_device)

    if precision == torch.int8:
        if qat:
            print('Calibrating with pytorch-quantization...')
            txt_filename = f'out/{model_name}_{precision}_bs{max_batch_size}_qat.txt'
            qat_model = copy.deepcopy(model_ft)
            quant_nn.TensorQuantizer.use_fb_fake_quant = True
            with torch.no_grad():
                calibrate_model(model=qat_model, data_loader=cal_dataloader['cal'])

            qat_dataloader = load_data(resolution, channels, ['train'], 'resources/dataset_qat_2001.csv', path=img_root, batch_size=64, test_device=test_device)
            dataloaders = load_data(resolution, channels, ['val'], 'resources/dataset_2001.csv', path=img_root, batch_size=64, test_device=test_device)

            optimizer = SGD(qat_model.parameters(), lr=0.0001, momentum=0)
            criterion = nn.CrossEntropyLoss()
            scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

            dataloaders.update(qat_dataloader)

            best_acc = 0
            print('Training...')
            with open(txt_filename, 'w') as sys.stdout:
                for nepoch in range(6):
                    print(f'Epoch {nepoch}/5')
                    print('-' * 10)

                    train_one_epoch(qat_model, criterion, optimizer, scheduler, dataloaders['train'], torch.device('cuda'))

                    qat_model.eval()
                    acc = validation(qat_model, criterion, optimizer, dataloaders['val'], torch.device('cuda'))

                    if acc > best_acc:
                        best_acc = acc
                        print(f' Best Acc: {best_acc:.4f}')
                        best_model = copy.deepcopy(qat_model)
            
            sys.stdout = sys.__stdout__
        
            best_model.eval()
            images, _ = next(iter(cal_dataloader['cal']))
            jit_model = torch.jit.trace(qat_model, images.to('cuda'))
            trt_ts_module = compile_trt_model(jit_model, cal_dataloader['cal'], precision, max_batch_size, len(channels), resolution, calibrate=False)
        else:
            quant_nn.TensorQuantizer.use_fb_fake_quant = True
            cal_model = copy.deepcopy(model_ft)
            with torch.no_grad():
                calibrate_model(model=cal_model, data_loader=cal_dataloader['cal'])
            images, _ = next(iter(cal_dataloader['cal']))
            jit_model = torch.jit.trace(cal_model, images.to('cuda'))
            trt_ts_module = compile_trt_model(jit_model, cal_dataloader['cal'], precision, max_batch_size, len(channels), resolution, calibrate=False)
    else:
        images, _ = next(iter(cal_dataloader['cal']))
        jit_model = torch.jit.trace(model_ft, images.to('cuda'))
        trt_ts_module = compile_trt_model(jit_model, cal_dataloader['cal'], precision, max_batch_size, len(channels), resolution, calibrate=False)
        
    for bsize in [pow(2, x) for x in range(20)]:
        test_dataloader = load_data(resolution, channels, ['test'], 'resources/dataset_2001.csv', path=img_root, class_path='resources/labels.txt', batch_size=bsize, test_device=test_device)['test']

        print('Warming up...')
        warm_up(trt_ts_module, 64, len(channels), resolution)
        
        running_corrects = 0.0
        running_time = 0.0
        total = 0.0
        for input_data, labels in test_dataloader:
            input_data = input_data.to(device)
            labels = labels.to(device)
            t1 = time.time()
            result = trt_ts_module(input_data)
            running_time += time.time() - t1
            _, preds = torch.max(result, 1)
            running_corrects += torch.sum(preds == labels.data)
            total += input_data.size(0)

        acc = running_corrects / total
        inference_speed = 1 / (running_time / total)
        file = open(filename, 'a')
        file.write(f'{bsize}, {acc:.4f}, {inference_speed:.4f}, {total:.0f}\n')
        file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--channels', nargs="+", type=str, default=['red', 'green', 'blue', 'nir', 'red_edge'])
    parser.add_argument('--img-root', type=str, default='../data')
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--resolution', type=int)
    parser.add_argument('--precision', type=str, default='float32')
    parser.add_argument('--test-device', type=bool)
    parser.add_argument('--qat', type=bool)
    args = parser.parse_args()
    precision = parse_precision(args.precision)
    main(args.channels, args.img_root, args.resolution, args.model_path, precision, args.test_device, args.qat)