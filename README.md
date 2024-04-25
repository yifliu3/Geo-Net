# Geo-Net: geometry-guided pre-training for tooth point cloud segmentation
This repository is an official PyTorch implementation of the paper **"Geo-Net: geometry-guided pre-training for tooth point cloud segmentation"**


## Environmental setup
Clone this repository into any place you want.
```bash
git clone https://github.com/CUHK-AIM-Group/UNSAM.git
cd UNSAM
mkdir data; mkdir pretrain;
```
## Pre-training 
* Train the UN-SAM with the default settings:
```python
python train.py --dataset data/$YOUR DATASET NAME$ --sam_pretrain pretrain/$SAM CHECKPOINT$
```

## Fine-tuning 
* Train the UN-SAM with the default settings:
```python
python train.py --dataset data/$YOUR DATASET NAME$ --sam_pretrain pretrain/$SAM CHECKPOINT$

```

## Evaluation 
* Train the UN-SAM with the default settings:
```python
python train.py --dataset data/$YOUR DATASET NAME$ --sam_pretrain pretrain/$SAM CHECKPOINT$

```

## Cite
If you find our work useful in your research or publication, please cite our work
```


## Acknowledgements
* [Point-MAE](https://github.com/Pang-Yatian/Point-MAE)
