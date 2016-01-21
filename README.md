# CNNMRF
This is the torch implementation for paper "Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis"

This algorithm is for
* un-guided image synthesis (for example, classical texture synthesis)
* guided image synthesis (for example, transfer the style between different images)

# Example
* un-guided image synthesis


# Setup

This code is based on Torch. It has only been tested on Mac and Ubuntu.

Dependencies:
* [Torch](https://github.com/torch/torch7)
* [loadcaffe](https://github.com/szagoruyko/loadcaffe)
* [CUDA](https://developer.nvidia.com/cuda-downloads)
* [cudnn](https://developer.nvidia.com/cudnn)

Pre-trained network:
We use the the original VGG-19 model. You can find the download script at [Neural Style](https://github.com/jcjohnson/neural-style). The downloaded model and prototxt file MUST be saved in the folder "data/models"

# Un-guided Synthesis
* Run `qlua run_syn.lua` in a terminal. The algorithm will create a synthesis image of twice the size as the style input image.
* The content/style images are located in the folders "data/content" and "data/style" respectively. Notice by default the content image is the same as the style image; and the content image is only used for initalization (optional). 
* Results are located in the folder "data/result/freesyn/MRF"
* Parameters are defined & explained in "run_syn.lua".

# Guided Synthesis
* Run `qlua run_trans.lua` in a terminal. The algorithm will synthesis using the texture of the style image and the structure of the content image. 
* The content/style images are located in the folders "data/content" and "data/style" respectively. 
* Results are located in the folder "data/result/trans/MRF"
* Parameters are defined & explained in "run_trans.lua".

# Hardware
* Our algorithm requires efficient GPU memory to run. A Titan X (12G memory) is able to complete the above examples with default setting. For GPU with 4G memory or 2G memory, please use the reference parameter setting in the  "run_trans.lua" and "run_syn.lua"

# Acknowledgement
* This work is inspired and closely related to the paper: [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576) by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge. The key difference between their method and our method is the different "style" constraints: While Gatys et al used a global constraint for non-photorealistic synthesis, we use a local constraint which works for both non-photorealistic and photorealistic synthesis. See our paper for more details.
* Our implementation is based on Justin Johnson's implementation of [Neural Style](https://github.com/jcjohnson/neural-style).   


