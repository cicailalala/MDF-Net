# MDF-Net: A Multi-Scale Dynamic Fusion Network for Breast Tumor Segmentation of Ultrasound Images

This repository is the official implementation of **MDF-Net: A Multi-Scale Dynamic Fusion Network for Breast Tumor Segmentation of Ultrasound Images, TIP2023** [Paper](https://ieeexplore.ieee.org/document/10232957).

# Requirements:
- Python 3.7
- Pytorch 1.7.0

# Datasets
Please download the dataset through [link](https://drive.google.com/file/d/1lhviQEuN537AzI6M5FNFuIBCK9AW2goG/view?usp=sharing). 

The project should be finally organized as follows:
```
./MDFNet/
  ├── data/
      ├── BUS_A/
      ├── BUS_B/
  ├── losses/
  ├── model/
  ├── dataset.py 
  ├── main.py
  ...... 
```

# Running
```
python main.py --dataset_name BUS_A --model MDFNet --img_size 320 --save_path ./model/
```

## Citations

```bibtex
@ARTICLE{10232957,
  author={Qi, Wenbo and Wu, H. C. and Chan, S. C.},
  journal={IEEE Transactions on Image Processing}, 
  title={MDF-Net: A Multi-Scale Dynamic Fusion Network for Breast Tumor Segmentation of Ultrasound Images}, 
  year={2023},
  volume={32},
  number={},
  pages={4842-4855},
  doi={10.1109/TIP.2023.3304518}}

```
