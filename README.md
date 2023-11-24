# STDFormer
STDFormer: Spatio Temporal Disentanglement Learning for 3D Human Mesh Recovery from Monocular Videos with Transformer
# Abstract
A novel spatio-temporal disentanglement method, STDFormer is presented, specifically designed for reconstructing sequential 3D human meshes from monocular videos. Precise and stable dynamic meshes are recovered, significantly reducing the phenomenon of human mesh distortion and mesh-vertex jitter. STDFormer for the first time adopts a vertex-based paradigm, featuring two main innovative points: the \textbf{spatial disentanglement (SD)} and the \textbf{temproal disentanglement (TD)}. The former is dedicated to extracting precise target features from coupled spatial information in frames, with a particular focus on feature extraction in complex backgrounds, and the latter, through the integration of temporal information across frames, effectively disentangles features in both spatial and temporal dimensions. The process mitigates estimation errors in inter-frame target features, ensuring highly accurate and motion-consistent reconstruction of human motion features in videos.  In comparisons of the SOTA
performance on the 3DPW benchmark, our experimental results
demonstrate that STDFormer effectively alleviates the issue
of mesh smoothness in frames while enhancing the accuracy of human motion estimation
