![]()

# OCR-SAM

## 🐇 Introduction 🐙
This repository is mainly to combine the TextDetector, TextRecgonizer， [Segment Anything](https://github.com/facebookresearch/segment-anything) and other Adavanced Tech to develop some **OCR-related Application Demo**.  
*Note: We will continue to update and maintain this repo, and develop more OCR-related advanced applications demo to the community. **Welcome anyones to join who have the idea and contribute to our repo**.*

## 📅 Updates 👀
- **2023.04.12**: Repository Release
- **2023.04.12**: Supported the [Inpainting](README.md#🏃🏻‍♂️-Run-Demo#Inpainting) combined with DBNet++, SAM and ControlNet.
- **2023.04.11**: Supported the [Erasing](README.md#🏃🏻‍♂️-Run-Demo#Erasing) combined with DBNet++, SAM and Latent-Diffusion / Stable-Diffusion.

## 📸 Demo Zoo 🔥

This project includes:

- [x] [Erasing](README.md#🏃🏻‍♂️-Run-Demo#Erasing): DBNet++ + SAM + Latent-Diffusion / Stable Diffusion 
![](./imgs/erase_vis.png)
- [x] [Inpainting](README.md#🏃🏻‍♂️-Run-Demo#Inpainting)


## 🚧 Installation 🛠️
### Prerequisites

- Linux | Windows
- Python 3.7
- Pytorch 1.6 or higher
- CUDA 11.3

### Environment Setup
Clone this repo:
```
git clone https://github.com/yeungchenwa/OCR-SAM.git
```
**Step 0**: Create a conda environment and activate it.
```
conda create --n ocr-sam python=3.8 -y
conda activate ocr-sam
```
**Step 1**: Install related version Pytorch following [here](https://pytorch.org/get-started/previous-versions/)

**Step 2**: Install the mmengine, mmcv, mmdet, mmcls, mmocr.
```
pip install -U openmim
mim install mmengine
mim install 'mmcv==2.0.0rc4'
mim install 'mmdet==3.0.0rc5'
mim install 'mmcls==1.0.0rc5'

# Install the mmocr from source
cd OCR-SAM
pip install -v -e .
```

**Step 3**: Prepare for the diffusers and latent-diffusion
```
# Install the diffusers
pip install diffusers

# Install the pytorch_lightning for ldm
conda install pytorch-lightning -c conda-forge
```


## 🏃🏻‍♂️ Run Demo 🏊‍♂️

### **Erasing**

### **Inpainting**


## 💗 Acknowledgement
- [segment-anything](https://github.com/facebookresearch/segment-anything)
- [latent-diffusion](https://github.com/CompVis/latent-diffusion)
- [mmocr](https://github.com/open-mmlab/mmocr)