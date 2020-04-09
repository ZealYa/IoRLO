# IoRLO: Intent-oriented Offloading algorithm

## Installation

This code is based on Python 3.5.2 and tensorflow with CUDA-10.1.

```
pip install -r requirements.txt
```

## Available Scenarios

### MEC-based scenarios

- IoRLO_AlexNet_MEC

- IoRLO_VGGNet-16_MEC

- IoRLO_ResNet-32_MEC

- IoRLO_MobileNet-V1_MEC

### MEC-based scenarios


- IoRLO_AlexNet_D2D

- IoRLO_VGGNet-16_D2D

- IoRLO_ResNet-32_D2D

- IoRLO_MobileNet-V1_D2D

## HOW TO

`$ python IoRLO_main.py`


## Note

- For MEC-based scenarios, (0.2, 0.6 / 0.25, 0.6 / 0.25, 0.7) is setted in the `IoRLO_main.py`

- For D2D-based scenarios, (0.4 / 0.6) is setted in the IoRLO_main.py, the (Co_H / Co_L) is setted in the `IoRLO.py`
