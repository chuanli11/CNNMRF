require 'torch'
require 'nn'
require 'image'
require 'paths'

paths.dofile('mylib/myoptimizer.lua')
paths.dofile('mylib/tv.lua')
paths.dofile('mylib/style.lua')
paths.dofile('mylib/content.lua')
paths.dofile('mylib/mrf.lua')

transfer_CNNMRF_wrapper = require 'transfer_CNNMRF_wrapper'

-- list of parameters:
-- content_name, 
-- style_name, 
-- ini_method, 
-- num_iter, 
-- mrf_layers, 
-- mrf_patch_size, 
-- mrf_num_rotation, 
-- mrf_num_scale, 
-- content_weight, 
local list_params = { 
                     {'0', '0', 'image', {400, 400, 400, 200}, {21}, {3}, 3, 3, 2e0},
                     {'1', '1', 'image', {400, 400, 400, 200}, {21}, {3}, 3, 3, 2e0},
                    }    

for i_test = 1, #list_params do
    local state = transfer_CNNMRF_wrapper.state(list_params[i_test][1], list_params[i_test][2], list_params[i_test][3], list_params[i_test][4], list_params[i_test][5], list_params[i_test][6], list_params[i_test][7], list_params[i_test][8], list_params[i_test][9])
end


do return end
