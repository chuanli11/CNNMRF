require 'torch'
require 'nn'
require 'image'
require 'paths'
require 'loadcaffe'

torch.setdefaulttensortype('torch.FloatTensor') -- float as default tensor type

local function run_test(content_name, style_name, ini_method, max_size, num_res, num_iter, mrf_layers, mrf_weight, mrf_patch_size, mrf_num_rotation, mrf_num_scale, mrf_sample_stride, mrf_synthesis_stride, mrf_confidence_threshold, content_layers, content_weight, tv_weight, mode, gpu_chunck_size_1, gpu_chunck_size_2)
  -- local clock = os.clock
  -- function sleep(n)  -- seconds
  --   local t0 = clock()
  --   while clock() - t0 <= n do end
  -- end

  local timer_TEST = torch.Timer()

  local flag_state = 1

  local cmd = torch.CmdLine()
  local params = cmd:parse(arg)

  -- externally set paramters
  params.content_name = content_name
  params.style_name = style_name
  params.ini_method = ini_method
  params.max_size = max_size or 384
  params.num_res = num_res or 3
  params.num_iter = num_iter or {100, 100, 100}
  params.mrf_layers = mrf_layers or {12, 21}
  params.mrf_weight = mrf_weight or {1e-4, 1e-4}
  params.mrf_patch_size = mrf_patch_size or {3, 3}
  params.target_num_rotation = mrf_num_rotation or 0
  params.target_num_scale = mrf_num_scale or 0
  params.target_sample_stride = mrf_sample_stride or {2, 2}
  params.source_sample_stride = mrf_synthesis_stride or {2, 2}
  params.mrf_confidence_threshold = mrf_confidence_threshold or {0, 0}
  params.content_layers = content_layers or {21}
  params.content_weight = content_weight or 2e1
  params.tv_weight = tv_weight or 1e-3

  params.mode = mode or 'speed'
  params.gpu_chunck_size_1 = gpu_chunck_size_1 or 256
  params.gpu_chunck_size_2 = gpu_chunck_size_2 or 16

  -- fixed parameters
  params.target_step_rotation = math.pi/24
  params.target_step_scale = 1.05
  os.execute('mkdir ' .. 'data/result/')
  os.execute('mkdir ' .. 'data/result/trans/')
  os.execute('mkdir ' .. 'data/result/trans/MRF/')
  params.output_folder = 'data/result/trans/MRF/' .. params.content_name .. '_TO_' .. params.style_name
  params.proto_file = 'data/models/VGG_ILSVRC_19_layers_deploy.prototxt'
  params.model_file = 'data/models/VGG_ILSVRC_19_layers.caffemodel'
  params.gpu = 0
  params.backend = 'cudnn'
  params.nCorrection = 25
  params.print_iter = 10
  params.save_iter = 10


  os.execute('mkdir ' .. params.output_folder)

  local function main(params)
    local net = nn.Sequential()
    local next_content_idx = 1
    local i_net_layer = 0
    local num_calls = 0
    local content_losses = {}
    local content_layers = {}
    local i_content_layer = 0
    local next_mrf_idx = 1
    local mrf_losses = {}
    local mrf_layers = {}
    local i_mrf_layer = 0
    local input_image
    local output_image
    local cur_res
    local content_layers_pretrained = params.content_layers
    local mrf_layers_pretrained = params.mrf_layers

    -----------------------------------------------------------------------------------
    -- read images
    -----------------------------------------------------------------------------------
    local source_image = image.load('data/content/' .. params.content_name  .. '.jpg', 3)
    local target_image = image.load('data/style/' .. params.style_name  .. '.jpg', 3)

    source_image = image.scale(source_image, params.max_size, 'bilinear')
    target_image = image.scale(target_image, params.max_size, 'bilinear')

    local render_height = source_image:size()[2]
    local render_width = source_image:size()[3]
    source_image_caffe = preprocess(source_image):float()
    target_image_caffe = preprocess(target_image):float()

    local pyramid_source_image_caffe = {}
    for i_res = 1, params.num_res do
      pyramid_source_image_caffe[i_res] = image.scale(source_image_caffe, math.ceil(source_image:size()[3] * math.pow(0.5, params.num_res - i_res)), math.ceil(source_image:size()[2] * math.pow(0.5, params.num_res - i_res)), 'bilinear')
    end

    local pyramid_target_image_caffe = {}
    for i_res = 1, params.num_res do
      pyramid_target_image_caffe[i_res] = image.scale(target_image_caffe, math.ceil(target_image:size()[3] * math.pow(0.5, params.num_res - i_res)), math.ceil(target_image:size()[2] * math.pow(0.5, params.num_res - i_res)), 'bilinear')
    end

    ------------------------------------------------------------------------------------------------------
    -- local function for adding a content layer
    ------------------------------------------------------------------------------------------------------
    local function add_content()
      local source =  pyramid_source_image_caffe[cur_res]:clone()
      if params.gpu >= 0 then
        source = source:cuda()
      end
      local feature = net:forward(source):clone() -- generate the content target using content image
      if params.gpu >= 0 then
        feature = feature:cuda()
      end

      local norm = params.normalize_gradients
      local loss_module = nn.ContentLoss(params.content_weight, feature, norm):float()
      if params.gpu >= 0 then
        loss_module:cuda()
      end

      i_content_layer = i_content_layer + 1
      i_net_layer = i_net_layer + 1
      next_content_idx = next_content_idx + 1 
      net:add(loss_module)
      table.insert(content_losses, loss_module)
      table.insert(content_layers, i_content_layer, i_net_layer)
    end

    local function update_content(idx_layer, idx_content)
      local source =  pyramid_source_image_caffe[cur_res]:clone()
      if params.gpu >= 0 then
        source = source:cuda()
      end
      net:forward(source)
      local feature = net:get(idx_layer).output:clone()
      if params.gpu >= 0 then
        feature = feature:cuda()
      end

      local norm = params.normalize_gradients
      local loss_module = nn.ContentLoss(params.content_weight, feature, norm):float()
      if params.gpu >= 0 then
        loss_module:cuda()
      end
      net:get(idx_layer):update(loss_module)
    end


    -- --------------------------------------------------------------------------------------------------------
    -- -- local function for adding a mrf layer, with image rotation andn scaling
    -- --------------------------------------------------------------------------------------------------------
    local function add_mrf()
        local mrf_module = nn.MRFMM()
        i_mrf_layer = i_mrf_layer + 1
        i_net_layer = i_net_layer + 1
        next_mrf_idx = next_mrf_idx + 1 
        if params.gpu >= 0 then
          mrf_module:cuda()
        end
        net:add(mrf_module)
        table.insert(mrf_losses, mrf_module)
        table.insert(mrf_layers, i_mrf_layer, i_net_layer)
        return true
    end

    local function build_mrf(id_mrf)
      --------------------------------------------------------
      -- deal with target
      --------------------------------------------------------
      target_images_caffe = {}
      for i_r = -params.target_num_rotation, params.target_num_rotation do
        local alpha = params.target_step_rotation * i_r 
        local min_x, min_y, max_x, max_y = computeBB(pyramid_target_image_caffe[cur_res]:size()[3], pyramid_target_image_caffe[cur_res]:size()[2], alpha)
        local target_image_rt_caffe = image.rotate(pyramid_target_image_caffe[cur_res], alpha, 'bilinear')
        target_image_rt_caffe = target_image_rt_caffe[{{1, target_image_rt_caffe:size()[1]}, {min_y, max_y}, {min_x, max_x}}]

        for i_s = -params.target_num_scale, params.target_num_scale do  
          local max_sz = math.floor(math.max(target_image_rt_caffe:size()[2], target_image_rt_caffe:size()[3]) * torch.pow(params.target_step_scale, i_s))
          local target_image_rt_s_caffe = image.scale(target_image_rt_caffe, max_sz, 'bilinear')
          target_image_rt_s_caffe = target_image_rt_s_caffe:cuda()
          table.insert(target_images_caffe, target_image_rt_s_caffe)
        end
      end

      -- compute the coordinates on the pixel layer
      local target_x
      local target_y
      local target_x_per_image = {}
      local target_y_per_image = {}
      local target_imageid
      -- print('*****************************************************')
      -- print(string.format('build target mrf'));
      -- print('*****************************************************')   
      for i_image = 1, #target_images_caffe do
        -- print(string.format('image %d, ', i_image))
        net:forward(target_images_caffe[i_image])
        local target_feature_map = net:get(mrf_layers[id_mrf] - 1).output:float()

        if params.mrf_patch_size[id_mrf] > target_feature_map:size()[2] or params.mrf_patch_size[id_mrf] > target_feature_map:size()[3] then 
          print('target_images is not big enough for patch')
          print('target_images size: ')
          print(target_feature_map:size())
          print('patch size: ')
          print(params.mrf_patch_size[id_mrf])
          do return end
        end 
        local target_x_, target_y_ = drill_computeMRFfull(target_feature_map,  params.mrf_patch_size[id_mrf], params.target_sample_stride[id_mrf], -1) 


        local x = torch.Tensor(target_x_:nElement() * target_y_:nElement())
        local y = torch.Tensor(target_x_:nElement() * target_y_:nElement())
        local target_imageid_ = torch.Tensor(target_x_:nElement() * target_y_:nElement()):fill(i_image)
        local count = 1
        for i_row = 1, target_y_:nElement() do
          for i_col = 1, target_x_:nElement() do
            x[count] = target_x_[i_col]
            y[count] = target_y_[i_row]
            count = count + 1
          end
        end
        if i_image == 1 then
          target_x = x:clone()
          target_y = y:clone()
          target_imageid = target_imageid_:clone()
        else
          target_x = torch.cat(target_x, x, 1)
          target_y = torch.cat(target_y, y, 1)
          target_imageid = torch.cat(target_imageid, target_imageid_, 1)
        end
        table.insert(target_x_per_image, x)
        table.insert(target_y_per_image, y)  
      end -- end for i_image = 1, #target_images do

      -- print('*****************************************************')
      -- print(string.format('collect mrf'));
      -- print('*****************************************************')  
      
      local num_channel_mrf = net:get(mrf_layers[id_mrf] - 1).output:size()[1]
      local target_mrf = torch.Tensor(target_x:nElement(), num_channel_mrf * params.mrf_patch_size[id_mrf] * params.mrf_patch_size[id_mrf])
      local tensor_target_mrf = torch.Tensor(target_x:nElement(), num_channel_mrf, params.mrf_patch_size[id_mrf], params.mrf_patch_size[id_mrf])
      local count_mrf = 1
      for i_image = 1, #target_images_caffe do
        -- print(string.format('image %d, ', i_image));
        net:forward(target_images_caffe[i_image])
        -- sample mrf on mrf_layers
        local tensor_target_mrf_, target_mrf_ = sampleMRFAndTensorfromLocation2(target_x_per_image[i_image], target_y_per_image[i_image], net:get(mrf_layers[id_mrf] - 1).output:float(), params.mrf_patch_size[id_mrf])        
        target_mrf[{{count_mrf, count_mrf + target_mrf_:size()[1] - 1}, {1, target_mrf:size()[2]}}] = target_mrf_:clone()
        tensor_target_mrf[{{count_mrf, count_mrf + target_mrf_:size()[1] - 1}, {1, tensor_target_mrf:size()[2]}, {1, tensor_target_mrf:size()[3]}, {1, tensor_target_mrf:size()[4]}}] = tensor_target_mrf_:clone()
        count_mrf = count_mrf + target_mrf_:size()[1]
        tensor_target_mrf_ = nil
        target_mrf_ = nil
        collectgarbage()
      end --for i_image = 1, #target_images do
      local target_mrfnorm = torch.sqrt(torch.sum(torch.cmul(target_mrf, target_mrf), 2)):resize(target_mrf:size()[1], 1, 1)

      --------------------------------------------------------
      -- process source
      --------------------------------------------------------
      -- print('*****************************************************')
      -- print(string.format('process source image'));
      -- print('*****************************************************')    
      net:forward(pyramid_source_image_caffe[cur_res]:cuda()) 
      local source_feature_map = net:get(mrf_layers[id_mrf] - 1).output:float()
      if params.mrf_patch_size[id_mrf] > source_feature_map:size()[2] or params.mrf_patch_size[id_mrf] > source_feature_map:size()[3] then 
        print('source_image_caffe is not big enough for patch')
        print('source_image_caffe size: ')
        print(source_feature_map:size())
        print('patch size: ')
        print(params.mrf_patch_size[id_mrf])
        do return end
      end 
      local source_xgrid, source_ygrid = drill_computeMRFfull(source_feature_map:float(), params.mrf_patch_size[id_mrf], params.source_sample_stride[id_mrf], -1) 
      local source_x = torch.Tensor(source_xgrid:nElement() * source_ygrid:nElement())
      local source_y = torch.Tensor(source_xgrid:nElement() * source_ygrid:nElement())      
      local count = 1
      for i_row = 1, source_ygrid:nElement() do
        for i_col = 1, source_xgrid:nElement() do
          source_x[count] = source_xgrid[i_col]
          source_y[count] = source_ygrid[i_row]
          count = count + 1
        end
      end
      -- local tensor_target_mrfnorm = torch.repeatTensor(target_mrfnorm:float(), 1, net:get(mrf_layers[id_mrf] - 1).output:size()[2] - (params.mrf_patch_size[id_mrf] - 1), net:get(mrf_layers[id_mrf] - 1).output:size()[3] - (params.mrf_patch_size[id_mrf] - 1)) 

      -- print('*****************************************************')
      -- print(string.format('call layer implemetation'));
      -- print('*****************************************************')  
      local nInputPlane = target_mrf:size()[2] / (params.mrf_patch_size[id_mrf] * params.mrf_patch_size[id_mrf])
      local nOutputPlane = target_mrf:size()[1]
      local kW = params.mrf_patch_size[id_mrf]
      local kH = params.mrf_patch_size[id_mrf]
      local dW = 1
      local dH = 1
      local input_size = source_feature_map:size()

      local source_xgrid_, source_ygrid_ = drill_computeMRFfull(source_feature_map:float(), params.mrf_patch_size[id_mrf], 1, -1) 
      local response_size = torch.LongStorage(3) 
      response_size[1] = nOutputPlane
      response_size[2] = source_ygrid_:nElement()
      response_size[3] = source_xgrid_:nElement()
      net:get(mrf_layers[id_mrf]):implement(params.mode, target_mrf, tensor_target_mrf, target_mrfnorm, source_x, source_y, input_size, response_size, nInputPlane, nOutputPlane, kW, kH, 1, 1, params.mrf_confidence_threshold[id_mrf], params.mrf_weight[id_mrf], params.gpu_chunck_size_1, params.gpu_chunck_size_2)
      target_mrf = nil
      tensor_target_mrf = nil
      source_feature_map = nil
      collectgarbage()
    end

    --------------------------------------------------------------------------------------------------------
    -- local function for printing inter-mediate result
    --------------------------------------------------------------------------------------------------------
    local function maybe_print(t, loss)
      local verbose = (params.print_iter > 0 and t % params.print_iter == 0)
      if verbose then
          print(string.format('Iteration %d, %d', t, params.num_iter[cur_res]))
       end
    end

    --------------------------------------------------------------------------------------------------------
    -- local function for saving inter-mediate result
    --------------------------------------------------------------------------------------------------------
    local function maybe_save(t)
      local should_save = params.save_iter > 0 and t % params.save_iter == 0
      should_save = should_save or t == params.num_iter
      if should_save then
      local disp = deprocess(input_image:float())
      disp = image.minmax{tensor=disp, min=0, max=1}
      disp = image.scale(disp, render_width, render_height, 'bilinear')
      filename = params.output_folder .. '/' .. 'res_' .. cur_res .. '_' .. t .. '.jpg'
      image.save(filename, disp)
      end
    end

    --------------------------------------------------------------------------------------------------------
    -- local function for computing energy
    --------------------------------------------------------------------------------------------------------      
    local function feval(x)
      num_calls = num_calls + 1
      net:forward(x)
      local grad = net:backward(x, dy)
      local loss = 0
      collectgarbage()

      maybe_print(num_calls, loss)
      maybe_save(num_calls)

      -- optim.lbfgs expects a vector for gradients
      return loss, grad:view(grad:nElement())
    end

    -------------------------------------------------------------------------------
    -- initialize network
    ------------------------------------------------------------------------------- 
    if params.gpu >= 0 then
      require 'cutorch'
      require 'cunn'
      cutorch.setDevice(params.gpu + 1)
    else
      params.backend = 'nn-cpu'
    end

    if params.backend == 'cudnn' then
      require 'cudnn'
    end

    local cnn = loadcaffe.load(params.proto_file, params.model_file, params.backend):float()
    if params.gpu >= 0 then
      cnn:cuda()
    end
    print('cnn succesfully loaded')

    for i_res = 1, params.num_res do
      local timer = torch.Timer()

      cur_res = i_res
      num_calls = 0
      local optim_state = {
        maxIter = params.num_iter[i_res],
        nCorrection = params.nCorrection,
        verbose=true,
        tolX = 0,
        tolFun = 0,
      }  

      -- initialize image and target
      if i_res == 1 then

        if params.ini_method == 'random' then
          input_image = torch.randn(pyramid_source_image_caffe[i_res]:size()):float():mul(0.001)
        elseif params.ini_method == 'image' then
          input_image = pyramid_source_image_caffe[i_res]:clone():float()
        else
          error('Invalid init type')
        end
        input_image = input_image:cuda()

        -----------------------------------------------------
        -- add a tv layer
        -----------------------------------------------------    
        if params.tv_weight > 0 then
          local tv_mod = nn.TVLoss(params.tv_weight):float()
          if params.gpu >= 0 then
            tv_mod:cuda()
          end
          i_net_layer = i_net_layer + 1
          net:add(tv_mod)
        end
        
        for i = 1, #cnn do
          if next_content_idx <= #content_layers_pretrained or next_mrf_idx <= #mrf_layers_pretrained then
            local layer = cnn:get(i)

            i_net_layer = i_net_layer + 1
            net:add(layer)

            -- add a content_losses layer
            if i == content_layers_pretrained[next_content_idx] then
              add_content()
            end

            -- -- add mrfstatsyn layer
            if i == mrf_layers_pretrained[next_mrf_idx] then
              if add_mrf() == false then
                print('build network failed: adding mrf layer failed')
                do return end
              end
            end

          end
        end -- for i = 1, #cnn do

        cnn = nil
        collectgarbage()
        
        print(net)
        
        print('content_layers: ')
        for i = 1, #content_layers do
          print(content_layers[i])
        end

        print('mrf_layers: ')
        for i = 1, #mrf_layers do
          print(mrf_layers[i])
        end

        print('network has been built.')
      else
        input_image = image.scale(input_image:float(), pyramid_source_image_caffe[i_res]:size()[3], pyramid_source_image_caffe[i_res]:size()[2], 'bilinear'):clone()
        input_image = input_image:cuda()           

        -- -- update content layers
        for i_layer = 1, #content_layers do
          update_content(content_layers[i_layer], i_layer)
          -- print(string.format('content_layers %d has been updated', content_layers[i_layer]))
        end

      end

      print('*****************************************************')
      print('Synthesis started at resolution ' .. cur_res)
      print('*****************************************************')

      print('Implementing mrf layers ...')
      for i = 1, #mrf_layers do
        if build_mrf(i) == false then
          print('build_mrf failed')
          do return end
        end
      end

      mask = torch.Tensor(input_image:size()):fill(1)
      mask = mask:cuda()
        
      y = net:forward(input_image)              
      dy = input_image.new(#y):zero()

      -- do optimizatoin
      local x, losses = mylbfgs(feval, input_image, optim_state, nil, mask) 

      local t = timer:time().real
      print('Synthesis finished at resolution ' .. cur_res ..  ', ' .. t .. ' seconds')
    end

  net = nil
  source_image = nil
  target_image = nil
  pyramid_source_image_caffe = nil
  pyramid_target_image_caffe = nil
  input_image = nil
  output_image = nil
  content_losses = nil
  content_layers = nil
  mrf_losses = nil
  mrf_layers = nil
  optim_state = nil
  collectgarbage()  
  collectgarbage()
      
  end -- end of main



  function preprocess(img)
    local mean_pixel = torch.Tensor({103.939, 116.779, 123.68})
    local perm = torch.LongTensor{3, 2, 1}
    img = img:index(1, perm):mul(256.0)
    mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
    img:add(-1, mean_pixel)
    return img
  end

  -- Undo the above preprocessing.
  function deprocess(img)
    local mean_pixel = torch.Tensor({103.939, 116.779, 123.68})
    mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
    img = img + mean_pixel:float()
    local perm = torch.LongTensor{3, 2, 1}
    img = img:index(1, perm):div(256.0)
    return img
  end
  
  main(params)

  local t_test = timer_TEST:time().real
  print('Total time:  ' .. t_test .. 'seconds.') 
  -- sleep(1)
  return flag_state
end

return {
  state = run_test
}
