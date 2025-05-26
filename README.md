# PG-SAG(PFG – Journal of Photogrammetry, Remote Sensing and Geoinformation Science)

PG-SAG: Parallel Gaussian Splatting for Fine-Grained Large-Scale Urban Buildings Reconstruction via Semantic-Aware Grouping

Tengfei Wang*, [Xin Wang*](https://xwangsgg.github.io/), Yongmao Hou, Yiwei Xu, Wendi Zhang, ZongqianZhan**.
![overall](https://github.com/user-attachments/assets/ed2369bc-398e-49db-9ebb-c2ad7d2f11bc)
### [Project Page]() | [paper](https://doi.org/10.1007/s41064-025-00343-0)

## Update
PG-SAG is accepted to PFG – Journal of Photogrammetry, Remote Sensing and Geoinformation Science. 
https://doi.org/10.1007/s41064-025-00343-0

## Installation


The repository contains submodules, thus please check it out with 
```shell
# SSH
git clone git@github.com:TFwang-9527/PG-SAG.git
cd PG-SAG

conda create -n pg-sag python=3.8
conda activate pg-sag

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 #replace your cuda version
pip install -r requirements.txt
pip install submodules/diff-plane-rasterization
pip install submodules/simple-knn
```
## Preprocessing
First,We need to obtain the building masks from the images. Here, we use lang-segment-anything (https://github.com/luca-medeiros/lang-segment-anything) for semantic segmentation.
After obtaining the masks, place them in the mask folder.

On the initial run of train.py, the building groups will be formed. Then, run run_vast_train.sh and run_vast_extract.sh to obtain the mesh for each building group.

The data folder should shown as follows
```shell
data
  ├── dataset
        ├── images
        ├── mask
        ├── sparse
```

## Training
python train.py -s data_path -m out_path --max_abs_split_points 0 --opacity_cull_threshold 0.05

## Rendering and Extract Mesh
python render.py -m out_path --max_depth 500.0 --voxel_size 0.01

## Acknowledgements
This project is built upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [PGSR](https://github.com/zju3dv/PGSR), [adrgaussian](https://github.com/hiroxzwang/adrgaussian) and [GauUscene](https://saliteta.github.io/CUHKSZ_SMBU/) . respectively. We thank all the authors for their great work and repos. 
