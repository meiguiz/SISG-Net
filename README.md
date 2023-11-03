SISG-Net:Simultaneous Instance Segmentation and Grasp Detection for Robot Grasp in Clutter

![图片18](https://github.com/meiguiz/SISG-Net/assets/90629126/9fcdc737-ca33-4899-bd70-f0eea9726160)

![4](https://github.com/meiguiz/SISG-Net/assets/90629126/45422f9a-5495-4f1e-a03e-2c2958ed1b43)
![5](https://github.com/meiguiz/SISG-Net/assets/90629126/c43d4be2-edf5-42f9-b2f4-a39c3a27bf01)
![6](https://github.com/meiguiz/SISG-Net/assets/90629126/6dd35863-37a4-44ab-b742-08cf11dc7ffd)



The video of our experiment used in this paper can be found at: https://youtu.be/G_8USOwFXVs


The relabled OCID dataset can be found at: https://drive.google.com/file/d/1-XBstG9ur90E4X66gdG_OLVidCR56pgV/view?usp=drive_link


1. Setup anaconda environment
```
python=3.7
torch=1.9.0
cuda=11.1
```
2. Set the path to the dataset in config file.
### Train

1. Download the OCID dataset at https://drive.google.com/file/d/1-XBstG9ur90E4X66gdG_OLVidCR56pgV/view?usp=drive_link.
2. Unzip the downloaded dataset, and modify the `dataset_path` of the config file correspondingly.
3. To train an SISG-Net on OCID dataset. 
```
$ python train.py --gpu 0 --cfg rgb_depth
