# torchscope
## 🔋 Energy Estimation Feature
This fork extends the original [torchscope](https://github.com/Tramac/torchscope) by adding **energy consumption analysis** for deep learning models, based on the sum of energy for FLOPs and energy for memory calculated as below:
- **FLOPs Energy**: `total_flops * 2.3 pJ` 
- **Memory Energy**: `total_memory * 640 pJ`

Note that the values `2.3pJ` and `640pJ` denotes the energy required to compute each FLOP and to store each MB, respectively. These values are obtained from [Han et al.](https://arxiv.org/abs/1602.01528).


### New Methods:
| Method                      | Description                                  |
|-----------------------------|--------------------------------------------|
| `get_layer_energy()`   | Returns energy (pJ) for a specific layer.  |
| `get_model_energy()`   | Returns total energy (pJ) for the model.   |


## Installation


- Install from source

```
$ pip install git+https://github.com/mzawishq/torchscope.git
```

## Usage

```python
from torchvision.models import resnet18
from torchscope import scope

model = resnet18()
scope(model, input_size=(3, 224, 224))
```

```
-------------------------------------------------------------------------------------------
        Layer (type)          Params           FLOPs    Memory (MBs)     Energy (pJ)
===========================================================================================
            Conv2d-1           9,408     118,013,952           12.36     271440000.00
       BatchNorm2d-2             128       1,605,632           12.25       3700794.50
              ReLU-3               0         802,816           12.25       1854316.80
         MaxPool2d-4               0         802,816            3.06       1848436.80
            Conv2d-5          36,864     115,605,504            3.48     265894880.00
       BatchNorm2d-6             128         401,408            3.06        925199.31
              ReLU-7               0         200,704            3.06        463579.20
            Conv2d-8          36,864     115,605,504            3.48     265894880.00
       BatchNorm2d-9             128         401,408            3.06        925199.31
             ReLU-10               0         200,704            3.06        463579.20
           Conv2d-11          36,864     115,605,504            3.48     265894880.00
      BatchNorm2d-12             128         401,408            3.06        925199.31
             ReLU-13               0         200,704            3.06        463579.20
           Conv2d-14          36,864     115,605,504            3.48     265894880.00
      BatchNorm2d-15             128         401,408            3.06        925199.31
             ReLU-16               0         200,704            3.06        463579.20
           Conv2d-17          73,728      57,802,752            2.38     132947848.00
      BatchNorm2d-18             256         200,704            1.53        462601.06
             ReLU-19               0         100,352            1.53        231789.60
           Conv2d-20         147,456     115,605,504            3.22     265894720.00
      BatchNorm2d-21             256         200,704            1.53        462601.06
           Conv2d-22           8,192       6,422,528            1.62      14772854.00
      BatchNorm2d-23             256         200,704            1.53        462601.06
             ReLU-24               0         100,352            1.53        231789.60
           Conv2d-25         147,456     115,605,504            3.22     265894720.00
      BatchNorm2d-26             256         200,704            1.53        462601.06
             ReLU-27               0         100,352            1.53        231789.60
           Conv2d-28         147,456     115,605,504            3.22     265894720.00
      BatchNorm2d-29             256         200,704            1.53        462601.06
             ReLU-30               0         100,352            1.53        231789.60
           Conv2d-31         294,912      57,802,752            4.14     132948976.00
      BatchNorm2d-32             512         100,352            0.77        231303.34
             ReLU-33               0          50,176            0.77        115894.80
           Conv2d-34         589,824     115,605,504            7.52     265897472.00
      BatchNorm2d-35             512         100,352            0.77        231303.34
           Conv2d-36          32,768       6,422,528            1.14      14772544.00
      BatchNorm2d-37             512         100,352            0.77        231303.34
             ReLU-38               0          50,176            0.77        115894.80
           Conv2d-39         589,824     115,605,504            7.52     265897472.00
      BatchNorm2d-40             512         100,352            0.77        231303.34
             ReLU-41               0          50,176            0.77        115894.80
           Conv2d-42         589,824     115,605,504            7.52     265897472.00
      BatchNorm2d-43             512         100,352            0.77        231303.34
             ReLU-44               0          50,176            0.77        115894.80
           Conv2d-45       1,179,648      57,802,752           13.88     132955216.00
      BatchNorm2d-46           1,024          50,176            0.39        115657.30
             ReLU-47               0          25,088            0.38         57947.40
           Conv2d-48       2,359,296     115,605,504           27.38     265910176.00
      BatchNorm2d-49           1,024          50,176            0.39        115657.30
           Conv2d-50         131,072       6,422,528            1.88      14773019.00
      BatchNorm2d-51           1,024          50,176            0.39        115657.30
             ReLU-52               0          25,088            0.38         57947.40
           Conv2d-53       2,359,296     115,605,504           27.38     265910176.00
      BatchNorm2d-54           1,024          50,176            0.39        115657.30
             ReLU-55               0          25,088            0.38         57947.40
           Conv2d-56       2,359,296     115,605,504           27.38     265910176.00
      BatchNorm2d-57           1,024          50,176            0.39        115657.30
             ReLU-58               0          25,088            0.38         57947.40
AdaptiveAvgPool2d-59               0               0            0.01             5.00
           Linear-60         513,000         512,000            5.89       1181367.12
===========================================================================================
Total params: 11,689,512
Trainable params: 11,689,512
Non-trainable params: 0
===========================================================================================
Total Giga-FLOPs (GFLOPs): 1.82
-------------------------------------------------------------------------------------------
Total Size (MBs): 248.45
-------------------------------------------------------------------------------------------
Total Energy (mJ): 4.19
-------------------------------------------------------------------------------------------
```

## Note

This plugin only supports the following operations:

-  Conv2d
- BatchNorm2d
- Pool2d
- ReLU
- Upsample

## Reference

- [pytorch-summary](https://github.com/sksq96/pytorch-summary)
- [torchstat](https://github.com/Swall0w/torchstat)