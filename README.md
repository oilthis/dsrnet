# DSRNet
Official PyTorch implementation of our paper (https://arxiv.org/abs/2407.19271)



## Dependencies
* python >= 3.5
* pytorch >= 1.1.0
* torchvision >= 0.4.0

## Prepare Dataset 
1. Download [GuanDao dataset](https://pan.baidu.com/s/1xiqy_0nZTvyM7MolYSaOug) password:kqfu
1. Place the GuanDao under dataset/
## Get Started
1. Clone this repo
    ```
    git clone https://github.com/oilthis/dsrnet.git
    cd DSRNet
    ```
1. Download the dataset. Modify the argument `--data_root` in `test.py` and `train.py` according to your data path.
### Evaluation

1. Download the [pretrained weights](https://pan.baidu.com/s/1zixwOyIQVx5ggAaX4uNTGQ) password:dq2r and place *.pth under pretrained_weights/

1. The testing results are in the `test_results/` folder

### Training

1. The training results are in the `weights/` folder

## Acknowledgement
This project is built on [MASA-SR](https://github.com/dvlab-research/MASA-SR). We thank the authors for their great work.
