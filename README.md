# Objective

This repo was designed as a improvement of the original [TDNet](https://github.com/feinanshan/TDNet/issues) code. 
The objective here is the creation of a Carla dataloader to allow train the pre-build Neural Network with CARLA Simulator images.

Some modifications to improve training speed were made inside the network desabling the teacher-student technique. 

## Installation:

#### Requirements:
1. Linux
2. Python 3.7
3. Pytorch 1.1.0
4. NVIDIA GPU + CUDA 10.0

#### Build

```bash
pip install -r requirements.txt
```

## Test with TDNet

see [TEST_README.md](./Testing/TEST_README.md)

## Train with TDNet

see [TRAIN_README.md](./Training/TRAIN_README.md)


## Citation
If you find TDNet is helpful in your research, please consider citing:

    @InProceedings{hu2020tdnet,
    title={Temporally Distributed Networks for Fast Video Semantic Segmentation},
    author={Hu, Ping and Caba, Fabian and Wang, Oliver and Lin, Zhe and Sclaroff, Stan and Perazzi, Federico},
    journal={CVPR},
    year={2020}
    }

## Disclaimer

This is a pytorch re-implementation of TDNet, please refer to the original paper [Temporally Distributed Networks for Fast Video Semantic Segmentation](http://openaccess.thecvf.com/content_CVPR_2020/papers/Hu_Temporally_Distributed_Networks_for_Fast_Video_Semantic_Segmentation_CVPR_2020_paper.pdf) for comparisons.

## References

- [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)
- [pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg)

## Observations

Original TDNet code: https://github.com/feinanshan/TDNet
