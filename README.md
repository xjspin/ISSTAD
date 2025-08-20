ISSTAD
======
![Image](https://github.com/xjspin/ISSTAD/raw/main/pngs/cover.png)
ISSTAD: Incremental Self-Supervised Learning Based on Transformer for Anomaly Detection and Localization

Introduction
-------------
This repository contains the source code for ISSTAD implemented on the MVTec AD dataset and the MVTec LOCO AD dataset.

If you are interested in evaluating ISSTAD on the **VisA** and **MPDD**  datasets, please refer to our extended implementation here:  
👉 https://github.com/xjpp2016/ISSTAD_Extended

Get Started
-------------
### Datasets
https://www.mvtec.com/company/research/datasets/mvtec-ad
https://www.mvtec.com/company/research/datasets/mvtec-loco

### Pre-trained MAE model
Kindly obtain the pre-trained MAE model from the provided link.  
https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth

### Environment
python==3.9.13  
matplotlib==3.6.0  
numpy==1.23.3  
opencv-python==4.6.0.66  
pandas==1.5.1  
pillow==9.2.0  
scikit-learn==1.1.2  
scipy==1.9.1  
six==1.16.0  
timm==0.3.2  
torch==1.12.1+cu116  
tqdm==4.64.1

The code is executable on Windows systems, and if running on Linux, it requires execution on a disk with an **NTFS**  file system. Otherwise, the results may degrade, especially for the localization result on the MVTec AD dataset.

### Run
MVTec AD dataset 
```bash
python main_mvtec_ad.py
```
MVTec LOCO AD dataset
```bash
python main_mvtec_loco_ad.py
```

### Citation
If you find this work helpful, please consider citing:

```
@article{JIN_ISSTAD,
title = {Incremental self-supervised learning based on transformer for anomaly detection and localization},
journal = {Engineering Applications of Artificial Intelligence},
volume = {160},
pages = {111978},
year = {2025},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2025.111978},
url = {https://www.sciencedirect.com/science/article/pii/S0952197625019864},
author = {Wenping Jin and Fei Guo and Qi Wu and Li Zhu},
keywords = {Anomaly detection, Vision transformer, Masked autoencoder, Pixel-level self-supervised learning},
}
```

```
@article{jin2023isstad,
  title={ISSTAD: Incremental self-supervised learning based on transformer for anomaly detection and localization},
  author={Jin, Wenping and Guo, Fei and Zhu, Li},
  journal={arXiv preprint arXiv:2303.17354},
  year={2023}
}
```