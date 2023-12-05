# STDFormer: Spatio-Temporal Disentanglement Learning for 3D Human Mesh Recovery from Monocular Videos with Transformer

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
### running
```bash
python /HOME/your......path/STDFormer/src/tools/run_STDFormer_bodymesh_dp_3dpw.py
```
Please download the STDFormer 3DPW Datatset weights we provide to /STDFormer/3dpw_checkpoint/state_dict.bin
[STDFormer_checkponit_3DPW](https://drive.google.com/file/d/1xiEAOaPhZyNI7M3xl3WnRJPnGF8Jn4rx/view?usp=sharing)
### change path
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
    parser.add_argument("--resume_checkpoint", default="/HOME/.........../STDFormer/3dpw_checkpoint/state_dict.bin", type=str, required=False,
                        help="Path to specific checkpoint for resume training.")
```
## Ablation experiment
![image](https://github.com/STDFormer-3D-Human-Mesh-Recovery/STDFormer/assets/121299261/45303a65-e97a-428f-affb-37d8680ae3e7)

Before the stage involving the participation of the transformer encoder and decoder in reconstruction, when extracting features from the target, one often encounters the following situations, especially in uncertain outdoor environments:

1. Multiple individuals appearing in the frame, interacting and correlating with the target person.
2. The foreground and background in the frame are highly complex, with the foreground obscuring the target object, and low distinguishability between the background and the target object.
3. For video tasks, the previously mentioned scenarios may involve motion, including the movement of the target object and the camera's own pose. Both stationary and moving non-target objects, or strong motion, can interfere with the extraction of target features. Particularly in video tasks, disturbances in motion features accumulate over the temporal axis, resulting in increased noise.

The above three points are categorized by us as spatiotemporal feature coupling, which is the primary reason for the inaccurate extraction of target features.

Therefore, the key to solving the feature coupling problem lies in decoupling the coupled features from both spatial and temporal perspectives to address the three points mentioned above.

#### Spatial Decoupling:

Spatial decoupling refers to, within a frame, using cross-channel attention learning to supervise target features and non-target features through different channel pooling and loss functions. After discretization, attention is concentrated on the channel where the target features are located, thereby enhancing the learning of target features within the frame and reducing attention to non-target features.

#### Temporal Decoupling:

Temporal decoupling involves forming a feature space based on sequential inputs, then learning the differences in features on the temporal sequence level through cross-channel learning. According to the attention weights of different channels, temporal decoupling is performed in the feature space to separate target features and non-target features on the temporal sequence level, which includes motion-related non-target features.

# License
This project is licensed under the terms of the MIT license.
