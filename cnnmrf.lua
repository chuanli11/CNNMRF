-- -*- coding: utf-8 -*-
require 'torch'
require 'paths'

paths.dofile('mylib/helper.lua')

-----------------------------------------
-- Parameters
-----------------------------------------



cmd = torch.CmdLine()

cmd:option('-content_name', 'potrait1', "The content image located in folder 'data/content'")
cmd:option('-style_name', 'picasso', "The style image located in folder 'data/style'")
cmd:option('-ini_method', 'image', "Initial method, set to 'image' to use the content image as the initialization; set to 'random' to use random noise.")
cmd:option('-type', 'transfer', 'transfer|syn')
cmd:option('-max_size',384, "Maximum size of the image. Larger image needs more time and memory.")
cmd:option('-backend','cudnn', "Use cudnn' for CUDA-enabled GPUs or 'clnn' for OpenCL.")
cmd:option('-mode','speed', "Try 'speed' if you have a GPU with more than 4GB memory, and try 'memory' otherwise. The 'speed' mode is significantly faster (especially for synthesizing high resolutions) at the cost of higher GPU memory. ")

cmd:option('-num_res',3, "Number of resolutions. Notice the lowest resolution image should be larger than the patch size otherwise it won't synthesize.")
cmd:option('-num_iter',{100, 100, 100}, "Number of iterations for each resolution.")
cmd:option('-mrf_layers',{12, 21}, "The layers for MRF constraint. Usualy layer 21 alone already gives decent results. Including layer 12 may improve the results but at significantly more computational cost.")
cmd:option('-mrf_weight',{1e-4, 1e-4}, "Weight for each MRF layer. Higher weights leads to more style faithful results.")
cmd:option('-mrf_patch_size',{3, 3}, "The patch size for MRF constraint. This value is defined seperately for each MRF layer.")
cmd:option('-target_num_rotation',0, 'To matching objects of different poses. This value is shared by all MRF layers. The total number of rotational copies is "2 * mrf_num_rotation + 1"')
cmd:option('-target_num_scale',0, 'To matching objects of different scales. This value is shared by all MRF layers. The total number of scaled copies is "2 * mrf_num_scale + 1"')
cmd:option('-target_sample_stride',{2, 2}, "Stride to sample mrf on style image. This value is defined seperately for each MRF layer.")
cmd:option('-mrf_confidence_threshold',{0, 0}, "Threshold for filtering out bad matching. Default value 0 means we keep all matchings. This value is defined seperately for all layers.")
cmd:option('-source_sample_stride',{2, 2}, "Stride to sample mrf on synthesis image. This value is defined seperately for each MRF layer. This settings is relevant only for syn setting.")

cmd:option('-content_layers',{21}, "The layers for content constraint")
cmd:option('-content_weight',2e1, "The weight for content constraint. Increasing this value will make the result more content faithful. Decreasing the value will make the method more style faithful. Notice this value should be increase (for example, doubled) if layer 12 is included for MRF constraint.")
cmd:option('-tv_weight',1e-3, "TV smoothness weight")
cmd:option('-scaler', 2, "Relative expansion from example to result. This settings is relevant only for syn setting.")

cmd:option('-gpu_chunck_size_1',256, "Size of chunks to split feature maps along the channel dimension. This is to save memory when normalizing the matching score in mrf layers. Use large value if you have large gpu memory. As reference we use 256 for Titan X, and 32 for Geforce GT750M 2G.")
cmd:option('-gpu_chunck_size_2',16, "Size of chuncks to split feature maps along the y dimension. This is to save memory when normalizing the matching score in mrf layers. Use large value if you have large gpu memory. As reference we use 16 for Titan X, and 2 for Geforce GT750M 2G.")

-- fixed parameters
cmd:option('-target_step_rotation', math.pi/24)
cmd:option('-target_step_scale', 1.05)
cmd:option('-output_folder', 'data/result/trans/MRF/')

cmd:option('-proto_file', 'data/models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'data/models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use')
cmd:option('-nCorrection', 25)
cmd:option('-print_iter', 10)
cmd:option('-save_iter', 10)

params = cmd:parse(arg)

local wrapper = nil
if params.type == 'transfer' then
    wrapper = require 'transfer_CNNMRF_wrapper'
else
    wrapper = require 'syn_CNNMRF_wrapper'
end

wrapper.main(params)