# CNNMRF
This is the torch implementation for paper "[Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis](http://arxiv.org/abs/1601.04589)"

This algorithm is for
* un-guided image synthesis (for example, classical texture synthesis)
* guided image synthesis (for example, transfer the style between different images)

# Hardware
* The algorithm requires a dedicate GPU. We provid two options here: choose 'speed' if your have at least 4GB graphic memory, and 'memory' otherwise. See our reference tests with Titan X and GT750M 2G in  "run_trans.lua" and "run_syn.lua"


# Examples
* guided image synthesis

<p><a href="/data/examples/content.jpg" target="_blank"><img src="/data/examples/content.jpg" height="320px" style="max-width:100%;"></a>
<a href="/data/examples/style.jpg" target="_blank"><img src="/data/examples/style.jpg" height="320px" style="max-width:100%;"></a>
<a href="/data/examples/Interpolation/3_balanced.png" target="_blank"><img src="/data/examples/Interpolation/3_balanced.png" height="320px" style="max-width:100%;"></a>
</p>

<p>A photo (left) is transfered into a painting (right) using Picasso's self portrait 1907 (middle) as the reference style. Notice important facial features, such as eyes and nose, are faithfully kept as those in the Picasso's painting.</p>

<p><a href="/data/content/1.jpg" target="_blank"><img src="/data/content/1.jpg" height="320px" style="max-width:100%;"></a>
<a href="/data/style/1.jpg" target="_blank"><img src="/data/style/1.jpg" height="320px" style="max-width:100%;"></a>
<a href="/data/examples/0_to_0.png" target="_blank"><img src="/data/examples/1_to_1.png" height="320px" style="max-width:100%;"></a></p>
<p>In this example, we first transfer a cartoon into a photo.</p>
<p><a href="/data/content/1.jpg" target="_blank"><img src="/data/style/1.jpg" height="320px" style="max-width:100%;"></a>
<a href="/data/style/1.jpg" target="_blank"><img src="/data/content/1.jpg" height="320px" style="max-width:100%;"></a>
<a href="/data/examples/1_to_1.png" target="_blank"><img src="/data/examples/0_to_0.png" height="320px" style="max-width:100%;"></a></p>
<p>We then swap the two inputs and transfer the photo into the cartoon.</p>

<p><a href="/data/examples/content.jpg" target="_blank"><img src="/data/examples/content.jpg" height="256px" style="max-width:100%;"></a>
<a href="/data/examples/Interpolation/2_morecontent.png" target="_blank"><img src="/data/examples/Interpolation/2_morecontent.png" height="256px" style="max-width:100%;"></a>
<a href="/data/examples/Interpolation/4_morestyle.png" target="_blank"><img src="/data/examples/Interpolation/4_morestyle.png" height="256px" style="max-width:100%;"></a>
<a href="/data/examples/style.jpg" target="_blank"><img src="/data/examples/style.jpg" height="256px" style="max-width:100%;"></a></p>
<p><a href="/data/examples/content2.jpg" target="_blank"><img src="/data/examples/content2.jpg" height="256px" style="max-width:100%;"></a>
<a href="/data/examples/Interpolation/2_morecontent2.png" target="_blank"><img src="/data/examples/Interpolation/2_morecontent2.png" height="256px" style="max-width:100%;"></a>
<a href="/data/examples/Interpolation/4_morestyle2.png" target="_blank"><img src="/data/examples/Interpolation/4_morestyle2.png" height="256px" style="max-width:100%;"></a>
<a href="/data/examples/style2.jpg" target="_blank"><img src="/data/examples/style2.jpg" height="200px" style="max-width:100%;"></a></p>
<p>It is possible to balance the amount of content and the style in the result: pictures in the second coloumn take more content, and pictures in the third column take more style.</p>

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

# Acknowledgement
* This work is inspired and closely related to the paper: [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576) by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge. The key difference between their method and our method is the different "style" constraints: While Gatys et al used a global constraint for non-photorealistic synthesis, we use a local constraint which works for both non-photorealistic and photorealistic synthesis. See our paper for more details.
* Our implementation is based on Justin Johnson's implementation of [Neural Style](https://github.com/jcjohnson/neural-style).   


