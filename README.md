# Transformer-based Variable-rate Image Compression With Region-of-interest Control
Accpeted to IEEE ICIP 2023

This repository contains the source code of our ICIP 2023 paper [arXiv](https://arxiv.org/abs/2306.05085).

## Abstract
>This paper proposes a transformer-based learned image compression system. It is capable of achieving variable-rate compression with a single model while supporting the regionof-interest (ROI) functionality. Inspired by prompt tuning, we introduce prompt generation networks to condition the transformer-based autoencoder of compression. Our prompt generation networks generate content-adaptive tokens according to the input image, an ROI mask, and a rate parameter. The separation of the ROI mask and the rate parameter allows an intuitive way to achieve variable-rate and ROI coding simultaneously. Extensive experiments validate the effectiveness of our proposed method and confirm its superiority over the other competing methods.

## Install
```bash
git clone https://github.com/NYCU-MAPL/Transformer_VariableROI
cd Transformer_VariableROI
pip install -U pip
pip install torch torchvision # have to match with the cuda version (we use 1.12.0+cu113)
pip install pillow==9.2.0
pip install shapely==1.7.1
pip install -e .
pip install timm tqdm click
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## Dataset
The following datasets are used and needed to be downloaded.
- Flicker2W (download [here](https://github.com/liujiaheng/CompressionData), and use [this script](https://github.com/xyq7/InvCompress/tree/main/codes/scripts) for preprocessing)
- COCO 2012 Train/Val
- Kodak

## Example Usage
Specify the data paths, target rate point, corresponding lambda, and checkpoint in the config file accordingly

### Training
We adopt three-stage training. 
1. Base codec pre-train for single rate: 
```bash
python examples/1_basecodec.py -c config/1_base_codec.yaml
```

2. Train for variable-rate without ROI present: 
```bash 
python examples/2_variablerate.py -c config/2_variablerate.yaml
```

3. Bring in ROI training: 
```bash 
python examples/3_ROI.py -c config/3_ROI.yaml
```

### Testing
Here shows example usage for two evaluation settings:
1. Variable-rate compression **without** ROI `(ROI:non-ROI=1:1)` on Kodak (Fig.3a in the paper)
```bash 
python examples/eval_variable.py -c config/eval.yaml
```

2. Variable-rate compression **with** ROI `(ROI:non-ROI=1:0)` on COCO Val (Fig.3b in the paper)
```bash
python examples/eval_ROI.py -c config/eval.yaml
```

By modifying the files `examples/eval_*.py`, the evaluation can be done in different ROI weighting or rate points

## Pre-trained Weights
The following released weight is for our proposed method on variable-rate with ROI functionalitiy. <br>
<a href="https://github.com/NYCU-MAPL/Transformer_VariableROI/releases/download/v1.0/transformer_variablerate_roi.pth.tar" download>
  download link
</a>

## Citation
If you find our project useful, please cite the following paper.
```
@inproceedings{kao2023transformer,
  title={Transformer-based Variable-rate Image Compression With Region-of-interest Control},
  author={Kao, Chia-Hao and Weng, Ying-Chieh and Chen, Yi-Hsin and Chiu, Wei-Chen and Peng, Wen-Hsiao},
  booktitle={Proceedings of the IEEE International Conference on Image Processing (ICIP)},
  pages={},
  year={2023}
}
```

## Ackownledgement
Our work is based on the framework of [CompressAI](https://github.com/InterDigitalInc/CompressAI). The base codec is adopted from [TIC](https://github.com/lumingzzz/TIC)/[TinyLIC](https://github.com/lumingzzz/TinyLIC) and the prompting method is modified from [VPT](https://github.com/KMnP/vpt). We thank the authors for open-sourcing their code.

