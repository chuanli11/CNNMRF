# CNNMRF
This is the code for paper "Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis"

This algorithm can be used for
* un-guided image synthesis (for example, classical texture synthesis)
* guided image synthesis (for example, transfer the style between different images)

# Setup

This code is based on Torch. Currently it is only tested on Mac and Ubuntu.

Dependencies:
* [Torch](https://github.com/torch/torch7)
* [loadcaffe](https://github.com/szagoruyko/loadcaffe)
* [CUDA](https://developer.nvidia.com/cuda-downloads)
* [cudnn](https://developer.nvidia.com/cudnn)

Pre-trained network:
We use the the original VGG-19 model. You can find the download script from [Neural Style](https://github.com/jcjohnson/neural-style). The downloaded model and prototxt file MUST be saved in "data/models"


# Style transfer 
* In terminal simply run `qlua run_trans.lua`
* The content/style images are located in "data/content" and "data/style" respectively. 
* Results are located in "data/result/tarns/MRF"
* Parameters are defined in "run_trans.lua"

# Un-guided Synthesis
* to be added

# Acknowledgement
* This work is inspired and closely related to the paper: [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576) by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge. The key difference between their method and our method is the different "style" constraints: While Gatys et al used a global constraint for non-photorealistic synthesis, we use a local constraint which works for both non-photorealistic and photorealistic synthesis. See our paper for more details.
* Our implementation is based on Justin Johnson's implementation of [Neural Style](https://github.com/jcjohnson/neural-style).   
