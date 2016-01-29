require 'torch'
require 'nn'
require 'image'
require 'paths'

paths.dofile('mylib/myoptimizer.lua')
paths.dofile('mylib/tv.lua')
paths.dofile('mylib/content.lua')
paths.dofile('mylib/mrf.lua')
paths.dofile('mylib/helper.lua')

transfer_CNNMRF_wrapper = require 'transfer_CNNMRF_wrapper'

-----------------------------------------
-- Parameters
-----------------------------------------
-- content_name: the content image located in folder "data/content"
-- style_name: the style image located in folder "data/style" 
-- ini_method: initial method, set to "image" to use the content image as the initialization; set to "random" to use random noise. 
-- max_size: maximum size of the synthesis image. Default value 384. Larger image needs more time and memory.
-- num_res: number of resolutions. Default value 3. Notice the lowest resolution image should be larger than the patch size otherwise it won't synthesize.
-- num_iter: number of iterations for each resolution. Default value 100 for all resolutions. 

-- mrf_layers: the layers for MRF constraint. Usualy layer 21 alone already gives decent results. Including layer 12 may improve the results but at significantly more computational cost.
-- mrf_weight: weight for each MRF layer. Default value 1e-4. Higher weights leads to more style faithful results.
-- mrf_patch_size: the patch size for MRF constraint. Default value 3. This value is defined seperately for each MRF layer.
-- mrf_num_rotation: To matching objects of different poses. Default value 0. This value is shared by all MRF layers. The total number of rotatoinal copies is "2 * mrf_num_rotation + 1"
-- mrf_num_scale: To matching objects of different scales. Default value 0. This value is shared by all MRF layers. The total number of scaled copies is "2 * mrf_num_scale + 1"
-- mrf_sample_stride: stride to sample mrf on style image. Default value 2. This value is defined seperately for each MRF layer.
-- mrf_synthesis_stride: stride to sample mrf on synthesis image. Default value 2. This value is defined seperately for each MRF layer.
-- mrf_confidence_threshold: threshold for filtering out bad matching. Default value 0 -- means we keep all matchings. This value is defined seperately for all layers.

-- content_layers: the layers for content constraint. Default value 23.
-- content_weight: The weight for content constraint. Default value 2e1. Increasing this value will make the result more content faithful. Decreasing the value will make the method more style faithful. Notice this value should be increase (for example, doubled) if layer 12 is included for MRF constraint,  

-- tv_weight: TV smoothness weight. Default value 1e-3.

-- mode: speed or memory. Try 'speed' if you have a GPU with more than 4GB memory, and try 'memory' otherwise. The 'speed' mode is significantly faster (especially for synthesizing high resolutions) at the cost of higher GPU memory.
-- gpu_chunck_size_1: Size of chunks to split feature maps along the channel dimension. This is to save memory when normalizing the matching score in mrf layers. Use large value if you have large gpu memory. As reference we use 256 for Titan X, and 32 for Geforce GT750M 2G.
-- gpu_chunck_size_2: Size of chuncks to split feature maps along the y dimension. This is to save memory when normalizing the matching score in mrf layers. Use large value if you have large gpu memory. As reference we use 16 for Titan X, and 2 for Geforce GT750M 2G.
-- backend: Use 'cudnn' for CUDA-enabled GPUs or 'clnn' for OpenCL.

-----------------------------------------
-- Reference tests 
-----------------------------------------
-- speed mode V.S. memory mode (Titan X 12G)
-- {'potrait1', 'picasso', 'image', 384, 3, {100, 100, 100}, {12, 21}, {1e-4, 1e-4}, {3, 3}, 1, 1, {2, 2}, {2, 2}, {0, 0}, {23}, 2e1, 1e-3, 'speed', 256, 16, 'cudnn'}, -- 101 seconds
-- {'potrait1', 'picasso', 'image', 384, 3, {100, 100, 100}, {12, 21}, {1e-4, 1e-4}, {3, 3}, 1, 1, {2, 2}, {2, 2}, {0, 0}, {23}, 2e1, 1e-3, 'memory', 256, 16, 'cudnn'}, -- 283 seconds

-- speed mode V.S. memory mode (Geforce GT750M 2G)
-- {'potrait1', 'picasso', 'image', 384, 3, {100, 100, 100}, {12, 21}, {1e-4, 1e-4}, {3, 3}, 1, 1, {2, 2}, {2, 2}, {0, 0}, {23}, 2e1, 1e-3, 'speed', 256, 16, 'cudnn'}, -- 570 seconds (gpu streching, not recommended) 
-- {'potrait1', 'picasso', 'image', 384, 3, {100, 100, 100}, {12, 21}, {1e-4, 1e-4}, {3, 3}, 1, 1, {2, 2}, {2, 2}, {0, 0}, {23}, 2e1, 1e-3, 'memory', 256, 16, 'cudnn'}, -- 973 seconds

-- speed mode V.S. memory mode (Sapphire Radeon R9 280 3G)
-- {'potrait1', 'picasso', 'image', 384, 3, {100, 100, 100}, {12, 21}, {1e-4, 1e-4}, {3, 3}, 1, 1, {2, 2}, {2, 2}, {0, 0}, {23}, 2e1, 1e-3, 'memory', 256, 16, 'clnn'}, -- 301 seconds (346 seconds total)

-- style interpolation (high resolution with Titan X 12G):
-- {'potrait1', 'picasso', 'image', 384, 3, {100, 100, 100}, {12, 21}, {1e-4, 1e-4}, {3, 3}, 1, 1, {2, 2}, {2, 2}, {0, 0}, {23}, 2e1, 1e-3, 'speed', 256, 16, 'cudnn'}, -- balanced                
-- {'potrait1', 'picasso', 'image', 384, 3, {100, 100, 100}, {12, 21}, {1e-4, 1e-4}, {3, 3}, 1, 1, {2, 2}, {2, 2}, {0, 0}, {23}, 4e1, 1e-3, 'speed', 256, 16, 'cudnn'}, -- more content 
-- {'potrait1', 'picasso', 'image', 384, 3, {100, 100, 100}, {12, 21}, {1e-4, 1e-4}, {3, 3}, 1, 1, {2, 2}, {2, 2}, {0, 0}, {23}, 1e1, 1e-3, 'speed', 256, 16, 'cudnn'}, -- more style 

-- style interpolation (low resolution with Geforce GT750M 2G):
-- {'potrait1', 'picasso', 'image', 256, 3, {100, 100, 100}, {12, 21}, {1e-4, 1e-4}, {3, 3}, 1, 1, {2, 2}, {2, 2}, {0, 0}, {23}, 2e1, 1e-3, 'speed', 32, 2, 'cudnn'},  -- balanced
-- {'potrait1', 'picasso', 'image', 256, 3, {100, 100, 100}, {12, 21}, {1e-4, 1e-4}, {3, 3}, 1, 1, {2, 2}, {2, 2}, {0, 0}, {23}, 4e1, 1e-3, 'speed', 32, 2, 'cudnn'}, -- more content 
-- {'potrait1', 'picasso', 'image', 256, 3, {100, 100, 100}, {12, 21}, {1e-4, 1e-4}, {3, 3}, 1, 1, {2, 2}, {2, 2}, {0, 0}, {23}, 1e1, 1e-3, 'speed', 32, 2, 'cudnn'}, -- more style 

-- other
-- {'0', '0', 'image', 384, 3, {100, 100, 100}, {12, 21}, {1e-4, 1e-4}, {3, 3}, 3, 3, {2, 2}, {2, 2}, {0, 0}, {23}, 2e1, 1e-3, 'speed', 256, 16, 'cudnn'}, -- Titan X 12G: 145 seconds
-- {'1', '1', 'image', 384, 3, {100, 100, 100}, {12, 21}, {1e-4, 1e-4}, {3, 3}, 3, 3, {2, 2}, {2, 2}, {0, 0}, {23}, 0.5e1, 1e-3, 'speed', 256, 16, 'cudnn'}, -- Titan X 12G: 146 seconds
-- {'0', '0', 'image', 256, 3, {100, 100, 100}, {12, 21}, {1e-4, 1e-4}, {3, 3}, 3, 3, {3, 3}, {2, 2}, {0, 0}, {23}, 1e1, 1e-3, 'speed', 32, 2, 'cudnn'}, -- Geforce GT750M 2G: 593 seconds
-- {'1', '1', 'image', 256, 3, {100, 100, 100}, {12, 21}, {1e-4, 1e-4}, {3, 3}, 3, 3, {3, 3}, {2, 2}, {0, 0}, {23}, 0.5e1, 1e-3, 'speed', 32, 2, 'cudnn'}, -- Geforce GT750M 2G: 623 seconds

   
local list_params = { 
                    {'potrait1', 'picasso', 'image', 128, 3, {100, 100, 100}, {12, 21}, {1e-4, 1e-4}, {3, 3}, 1, 1, {2, 2}, {2, 2}, {0, 0}, {23}, 2e1, 1e-3, 'speed', 256, 16, 'clnn'},
                    --{'potrait1', 'picasso', 'image', 384, 3, {100, 100, 100}, {12, 21}, {1e-4, 1e-4}, {3, 3}, 1, 1, {2, 2}, {2, 2}, {0, 0}, {23}, 2e1, 1e-3, 'speed', 256, 16, 'cudnn'},
                    --{'0', '0', 'image', 384, 3, {100, 100, 100}, {12, 21}, {1e-4, 1e-4}, {3, 3}, 3, 3, {2, 2}, {2, 2}, {0, 0}, {23}, 2e1, 1e-3, 'speed', 256, 16, 'cudnn'},
                    --{'1', '1', 'image', 384, 3, {100, 100, 100}, {12, 21}, {1e-4, 1e-4}, {3, 3}, 3, 3, {2, 2}, {2, 2}, {0, 0}, {23}, 0.5e1, 1e-3, 'speed', 256, 16, 'cudnn'},
                    }    

for i_test = 1, #list_params do
    local state = transfer_CNNMRF_wrapper.state(list_params[i_test][1], list_params[i_test][2], list_params[i_test][3], list_params[i_test][4], list_params[i_test][5], list_params[i_test][6], list_params[i_test][7], list_params[i_test][8], list_params[i_test][9], list_params[i_test][10], list_params[i_test][11], list_params[i_test][12], list_params[i_test][13], list_params[i_test][14], list_params[i_test][15], list_params[i_test][16], list_params[i_test][17], list_params[i_test][18], list_params[i_test][19], list_params[i_test][20], list_params[i_test][21])
    collectgarbage()
end


do return end
