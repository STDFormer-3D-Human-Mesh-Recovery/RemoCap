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
conda create --name PHMR python=3.8

# Activate conda environment
conda activate PHMR

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
python setup.py build develop

# Install requirements
pip install -r requirements.txt
pip install ./manopth/.
pip install --upgrade azureml-core


```
