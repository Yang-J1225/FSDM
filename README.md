# FSDM

## Requirements
* Python 3.9.16, Pytorch 1.12.1, [xformers](https://github.com/facebookresearch/xformers) 0.0.20
* More detail (See [environment.yml](environment.yml))
A suitable [conda](https://conda.io/) environment named `FSDM` can be created and activated with:

```
conda env create -f environment.yaml
conda activate FSDM
```

## Datasets
### REDS
用于训练和验证: https://seungjunnah.github.io/Datasets/reds.html

目前只需用到 train_sharp 和 val_sharp

## Configs

 ```configs/bicubic_swinunet_bicubic256.yaml```

根据实际情况修改 数据集和权重文件 的路径


## Inference

```
CUDA_VISIBLE_DEVICES=[GPU_ID] python inference_vsr.py --input_path [video_path] --out_path [VISUAL_RESULTS] --steps 15 --window_size 3
```

## Training
```
CUDA_VISIBLE_DEVICES=0  python main.py --cfg_path configs/bicubic_swinunet_bicubic256.yaml --save_dir work_dir  --steps 15
```