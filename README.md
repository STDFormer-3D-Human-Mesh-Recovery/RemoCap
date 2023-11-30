# STDFormer: Spatio Temporal Disentanglement Learning for 3D Human Mesh Recovery from Monocular Videos with Transformer

# Introduction
This is the official code repository for the Pytorch implementation of STDFormer
More details in [Project page](https://2573545674.wixsite.com/stdformer)
# Abstract
![pipline](https://github.com/STDFormer-3D-Human-Mesh-Recovery/STDFormer/assets/121299261/35d7707b-54f1-499f-8f44-c5b6300f6012)
A novel spatio-temporal disentanglement method, STDFormer is presented, specifically designed for reconstructing sequential 3D human meshes from monocular videos. Precise and stable dynamic meshes are recovered, significantly reducing the phenomenon of human mesh distortion and mesh-vertex jitter. STDFormer for the first time adopts a vertex-based paradigm, featuring two main innovative points: the spatial disentanglement (SD) and the temproal disentanglement (TD). The former is dedicated to extracting precise target features from coupled spatial information in frames, with a particular focus on feature extraction in complex backgrounds, and the latter, through the integration of temporal information across frames, effectively disentangles features in both spatial and temporal dimensions. The process mitigates estimation errors in inter-frame target features, ensuring highly accurate and motion-consistent reconstruction of human motion features in videos.  In comparisons of the SOTA
performance on the 3DPW benchmark, our experimental results
demonstrate that STDFormer effectively alleviates the issue
of mesh smoothness in frames while enhancing the accuracy of human motion estimation.

# Result

![ff963148652c68d6a8d21770c9dd530](https://github.com/STDFormer-3D-Human-Mesh-Recovery/STDFormer/assets/121299261/49424b81-0b5f-45ca-8aec-f4323f5a6e83)

The figure show the performance of our model in a sequential video task, showing the stability of the reconstruction results, the stability against occlusion interference and the advantages of limb position alignment.

Here we report the performance of STDFormer.
![image](https://github.com/STDFormer-3D-Human-Mesh-Recovery/STDFormer/assets/121299261/2cbeabbb-4f5a-45cb-bd1d-3a01b3a13cb9)

![image](https://github.com/STDFormer-3D-Human-Mesh-Recovery/STDFormer/assets/121299261/87e25816-fcde-41be-a033-162b96292b86)

# Running STDFormer
The base codes are largely borrowed from [FasterMETRO](https://github.com/postech-ami/FastMETRO) and [PointHMR](https://github.com/DCVL-3D/PointHMR_release).
## Installation
```bash
# We suggest to create a new conda environment with python version 3.8
conda create --name STDF python=3.8

# Activate conda environment
conda activate STDF

# Install Pytorch that is compatible with your CUDA version
# CUDA 10.1
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
# CUDA 10.2
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.2 -c pytorch
# CUDA 11.1
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge

# Install apex
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext
cd ..

# Install OpenDR
pip install matplotlib
pip install git+https://gitlab.eecs.umich.edu/ngv-python-modules/opendr.git

# Install STDFormer
git clone --recursive https://github.com/STDFormer-3D-Human-Mesh-Recovery/STDFormer.git
cd STDFormer
Unzip the file STDFormer/src/model.zip
python setup.py build develop

# Install requirements
pip install -r requirements(1).txt
pip install ./manopth/.
pip install --upgrade azureml-core


```
## Dataset
Please refer to [PointHMR](https://github.com/DCVL-3D/PointHMR_release). directly for dataset download and processing.

# Experiment
## Training

```bash
python /HOME/your......path/STDFormer/src/tools/run_STDFormer_bodymesh_dp_3dpw.py
```
```bash
#You need change yourself path

def parse_args():
    parser = argparse.ArgumentParser()
    #########################################################
    # Data related arguments
    #########################################################
    parser.add_argument("--data_dir", default='datasets', type=str, required=False,
                        help="Directory with all datasets, each in one subfolder")
    parser.add_argument("--train_yaml", default='/HOME/HOME/data/PointHMR/datasets/3dpw/train.yaml', type=str, required=False,
                        help="Yaml file with all data for training.")
    parser.add_argument("--val_yaml", default='/HOME/HOME/data/PointHMR/datasets/3dpw/test_has_gender.yaml', type=str, required=False,
                        help="Yaml file with all data for validation.")
 ..........................
..............................
    #########################################################
    # Loading/Saving checkpoints
    #########################################################
    parser.add_argument("--output_dir", default='/HOME/............./output_3DPWZ_result', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--saving_epochs", default=1, type=int)
    parser.add_argument("--resume_checkpoint", default="/HOME/.........../STDFormer/3dpw_checkpoint/checkpoint-5-880/state_dict.bin", type=str, required=False,
                        help="Path to specific checkpoint for resume training.")
```
# License
This project is licensed under the terms of the MIT license.
