# QuantizedBlackgrassClassification

A repository to experiment with classifying blackgrass with ResNet-18 in images with different levels of precision on different types of hardware.

## Dataset
The blackgrass dataset used here will be published soon...

## Spectral Channels
The implementation supports training a model on multispectral data: red, green, blue, near-infrared and red edge. Training and inference can be performed on a subset of channels by specifying the `--channels` parameter.

## TensorRT implementation
To accelerate GPU inference speed, there is a TensorRT implementation.  

To train the model:
```
python src/train.py --batch-size 64 --channels red green blue --img-root ../data resolution 512 --num-epochs 50
```

To test model on a GPU:
```
python src/test.py --model-path 'models/model.pth' --resolution 512
```
