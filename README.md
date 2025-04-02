# Geo-Net: geometry-guided pre-training for tooth point cloud segmentation
This repository is an official PyTorch implementation of the paper **"Geo-Net: geometry-guided pre-training for tooth point cloud segmentation"**

## Dataset
**Pre-training dataset:** 
You can download it from [onedrive](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155195605_link_cuhk_edu_hk/Ep1NAYTWMJdDhazoy-FggpoBNJozTtO8HVTrmCR8NTJgAw?e=rI9Hik).
After downloading, please modify the path in `cfgs/dataset_configs/Teethseg3D.yaml`.
Then, to get curvatures for the pre-training, run `python estim_curvs.py` (about 3 hours). 

**Fine-tuning dataset:**
Download link: https://osf.io/xctdy/.
After downloading, please modify the path in `cfgs/dataset_configs/Teethseg3D_finetune.yaml`.
## Dependencies
* Python 3.8.18
* Torch 1.13.1+cu117


## Setup
```bash
git clone https://github.com/yifliu3/Geo-Net.git
cd Geo-Net
conda create -n geonet python=3.8 -y
conda activate geonet
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
pip install -r requirements.txt
cd extensions/chamfer_dist && python setup.py install && cd ../..
cd extensions/pointops && python setup.py install && cd ../..
cd extensions/pointnet2 && python setup.py install && cd ../..
```


## Pre-training 
* Pre-train the Geo-Net with the default settings:
```python
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/pretrain_teethseg3d.yaml --exp_name pretrain_teethseg
```

## Fine-tuning 
* Fine-tune the Geo-Net with the default settings:
```python
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_teethseg3d.yaml --exp_name scratch_teethseg --val_freq 5 --finetune_model 
```

## Cite
If you find our work useful in your research or publication, please cite our work


## Acknowledgements
Some source code of ours is borrowed from [Point-MAE](https://github.com/Pang-Yatian/Point-MAE) and [MaskPoint](https://github.com/WisconsinAIVision/MaskPoint). Thanks for their contributions. 


