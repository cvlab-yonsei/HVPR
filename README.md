# PyTorch implementation of "HVPR: Hybrid Voxel-Point Representation for Single-stage 3D Object Detection"

<p align="center"><img src="./HVPR_files/overview.png" alt="no_image" width="40%" height="40%" /><img src="./HVPR_files/teaser.png" alt="no_image" width="60%" height="60%" /></p>
This is the implementation of the paper "HVPR: Hybrid Voxel-Point Representation for Single-stage 3D Object Detection (CVPR 2021)".

Our code is mainly based on [`OpenPCDet`](https://github.com/open-mmlab/OpenPCDet). We also plan to release the code based on [`PointPillars`](https://github.com/nutonomy/second.pytorch). 
For more information, checkout the project site [[website](https://cvlab.yonsei.ac.kr/projects/HVPR/)] and the paper [[PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Noh_HVPR_Hybrid_Voxel-Point_Representation_for_Single-Stage_3D_Object_Detection_CVPR_2021_paper.pdf)].

## Dependencies
* Python >= 3.6
* PyTorch >= 1.4.0

## Update
* 20/06/21: First update

## Installation
* Clone this repo, and follow the steps below (or you can follow the installation steps in [`OpenPCDet`](https://github.com/open-mmlab/OpenPCDet)).
1. Clone this repository:
   ```bash
   git clone https://github.com/cvlab-yonsei/HVPR.git
   ```
2. Install the dependent libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Install the `SparseConv` library from
[`spconv`](https://github.com/traveller59/spconv).

4. Install `pcdet` library:
   ```bash
   python setup.py develop
   ```
## Datasets
* KITTI 3D Object Detection 

1. Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows (the road planes could be downloaded from [[road plane]](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing), which are optional for data augmentation in the training):
     ```bash
     HVPR
     ├── data
     │   ├── kitti
     │   │   │── ImageSets
     │   │   │── training
     │   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes)
     │   │   │── testing
     │   │   │   ├──calib & velodyne & image_2
     ├── pcdet
     ├── tools
     ```
2. Generate the data infos by running the following command:
    ```bash
    python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
    ```
## Training
* The config files is in tools/cfgs/hvpr, and you can easily train your own model like:
  ```bash
  cd tools
  sh scripts/train_hvpr.sh 
  ```
* You can freely define parameters with your own settings like:
  ```bash
  cd tools
  sh scripts train_hvpr.sh --gpus 1 --result_path 'your_dataset_directory' --exp_dir 'your_log_directory'
  ```

## Evaluation
* Test your own model:
  ```bash
  cd tools
  sh scripts/eval_hvpr.sh
  ```


## Pre-trained model
* Download our pre-trained model.
<br>[[KITTI 3D Car]()]

## Bibtex
```
@inproceedings{park2020learning,
  title={Learning Memory-guided Normality for Anomaly Detection},
  author={Park, Hyunjong and Noh, Jongyoun and Ham, Bumsub},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14372--14381},
  year={2020}
}
```
## References
Our work is mainly built on [`OpenPCDet`](https://github.com/open-mmlab/OpenPCDet) codebase. Portions of our code are also borrowed from [`spconv`](https://github.com/traveller59/spconv), [`MemAE`](https://github.com/donggong1/memae-anomaly-detection), and [`CBAM`](https://github.com/Jongchan/attention-module). Thanks to the authors!
