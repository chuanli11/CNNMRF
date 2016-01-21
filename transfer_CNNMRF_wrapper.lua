require 'torch'
require 'nn'
require 'image'
require 'paths'

torch.setdefaulttensortype('torch.FloatTensor') -- float as default tensor type



local function run_test(content_name, style_name, ini_method, num_iter, mrf_layers, mrf_patch_size, mrf_num_rotation, mrf_num_scale, content_weight)

  

  local flag_state = 1

  local cmd = torch.CmdLine()

  local function main(params)

      

      local timer_main = torch.Timer()

      local ini_block_caffe = nil
      local coord_block_y = nil
      local coord_block_x = nil
      local content_layers_pretrained = params.content_layers_pretrained
      local style_layers_pretrained = params.style_layers_pretrained
      local style_layer_weights = params.style_layer_weights
      local mrf_layers_pretrained = params.mrf_layers_pretrained
      local mrf_layer_weights = params.mrf_layer_weights
      local mrf_layer_patch_size = params.mrf_layer_patch_size
      local mrf_layer_sample_stride = params.mrf_layer_sample_stride
      local mrf_layer_synthesis_stride = params.mrf_layer_synthesis_stride
      local mrf_layer_confidence_threshold = params.mrf_layer_confidence_threshold
      local content_layers = {}
      local i_content_layer = 0
      local next_content_idx = 1
      local style_layers = {}
      local i_style_layer = 0
      local next_style_idx = 1
      local mrf_layers = {}
      local i_mrf_layer = 0
      local next_mrf_idx = 1
      local content_losses, style_losses = {}, {}
      local input_caffe = nil
      local output_caffe = nil
      local output = nil 
      local num_calls = 0
      local y = nil
      local dy = nil
      local mask = nil
      local mask_weight = nil


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
      -----------------------------------------------------
      print('read & scale content and style images')
      -----------------------------------------------------
      local content_image = image.load(params.content_image, 3)
      content_image = image.scale(content_image, params.image_size, 'bilinear')
      local content_image_caffe = preprocess(content_image):float()
      print('content_image size: ')
      print(content_image:size())

      local style_image = image.load(params.style_image, 3)
      style_image = image.scale(style_image, params.style_size, 'bilinear')
      local style_image_caffe = preprocess(style_image):float()
      print('style_image size: ')
      print(style_image:size())

      local mrf_image = image.load(params.mrf_image, 3)
      mrf_image = image.scale(mrf_image, params.style_size, 'bilinear')
      print('mrf_image size: ')
      print(mrf_image:size())

      local ini_image = image.load(params.ini_image, 3)
      ini_image = image.scale(ini_image, content_image:size()[3], content_image:size()[2], 'bilinear')
      local ini_image_caffe = preprocess(ini_image):float()
      print('ini_image size: ')
      print(ini_image:size())   

      if params.gpu >= 0 then
        content_image_caffe = content_image_caffe:cuda()
        style_image_caffe = style_image_caffe:cuda()
        ini_image_caffe = ini_image_caffe:cuda()    
      end

      if params.content_block_flag == 1 then
        coord_block_x, coord_block_y = computegrid(content_image_caffe:size()[3], content_image_caffe:size()[2], params.content_block_size, params.content_block_stride, 1)
        print('coord_block_y: ')
        print(coord_block_y)  
        print('coord_block_x: ')
        print(coord_block_x)  
        ini_block_caffe = torch.randn(3, params.content_block_size, params.content_block_size):float():mul(0.001)
        if params.gpu >= 0 then
          ini_block_caffe = ini_block_caffe:cuda()
        end
      end

      -----------------------------------------------------
      print('Build network')
      -----------------------------------------------------
      local i_net_layer = 0
      local net = nn.Sequential()
   
      --------------------------------------------------------------------------------------------------------
      -- local function for adding a mrf layer, with image rotation andn scaling
      --------------------------------------------------------------------------------------------------------
      local function add_layer_mrf()


        -- do rotation 
        local filters_mrf = torch.Tensor(0, 0)
        local flag_first = 1
        for i_r = -params.mrf_num_rotation, params.mrf_num_rotation do
          local alpha = params.mrf_step_rotation * i_r 
          local min_x, min_y, max_x, max_y = computeBB(mrf_image:size()[3], mrf_image:size()[2], alpha)
          local mrf_image_rt = image.rotate(mrf_image, alpha, 'bilinear')
          mrf_image_rt = mrf_image_rt[{{1, mrf_image_rt:size()[1]}, {min_y, max_y}, {min_x, max_x}}]

          for i_s = -params.mrf_num_scale, params.mrf_num_scale do
            local max_sz = math.floor(math.max(mrf_image_rt:size()[2], mrf_image_rt:size()[3]) * torch.pow(params.mrf_step_scale, i_s))
            local mrf_image_rt_s = image.scale(mrf_image_rt, max_sz, 'bilinear')
            local mrf_image_caffe_rt = preprocess(mrf_image_rt_s):float()
            if params.gpu >= 0 then
              mrf_image_caffe_rt = mrf_image_caffe_rt:cuda()
            end
            -- forward the mrf image
            local target_features = net:forward(mrf_image_caffe_rt):clone()
            if mrf_layer_patch_size[next_mrf_idx] > target_features:size()[2] or mrf_layer_patch_size[next_mrf_idx] > target_features:size()[3] then 
              return false
            end

            -- design a bunch of filters
            local t_filters_mrf_, filters_mrf_, coord_x_, coord_y_ = computeMRF(target_features,  mrf_layer_patch_size[next_mrf_idx],  mrf_layer_sample_stride[next_mrf_idx], params.gpu) 
            mrf_num_x = coord_x_:nElement()
            mrf_num_y = coord_y_:nElement()

            -- CL: scale the filter by number of entries in the filter. Otherwise the convlution result will overshoot
            filters_mrf_ = filters_mrf_ * (1 / (mrf_layer_patch_size[next_mrf_idx] * mrf_layer_patch_size[next_mrf_idx] * target_features:size()[1]))
            if flag_first == 1 then
              filters_mrf = filters_mrf_:float():clone()
              flag_first = 0
            else
              filters_mrf = torch.cat(filters_mrf, filters_mrf_:float(), 1)
            end
            target_features = nil
            mrf_image_caffe_rt = nil             
            filters_mrf_ = nil
            collectgarbage()
          end -- for scale
        end -- for rotation     

        if params.gpu >= 0 then
          filters_mrf = filters_mrf:cuda()
        end

        -- make a mrf layer
        local nInputPlane = filters_mrf:size()[2] / (mrf_layer_patch_size[next_mrf_idx] * mrf_layer_patch_size[next_mrf_idx])
        local nOutputPlane = filters_mrf:size()[1]
        local kW = mrf_layer_patch_size[next_mrf_idx]
        local kH = mrf_layer_patch_size[next_mrf_idx]
        local dW = mrf_layer_synthesis_stride[next_mrf_idx]
        local dH = mrf_layer_synthesis_stride[next_mrf_idx]

        local mrf_module = nn.MRFMM(filters_mrf, nInputPlane, nOutputPlane, kW, kH, dW, dH, 0, 0, mrf_layer_weights[next_mrf_idx], mrf_layer_confidence_threshold[next_mrf_idx], params.gpu, mrf_num_x, mrf_num_y)
        filters_mrf = nil
        collectgarbage()
        if params.gpu >= 0 then
          mrf_module:cuda()
        else
          mrf_module:float()
        end

        i_mrf_layer = i_mrf_layer + 1
        table.insert(mrf_layers, i_mrf_layer, i_net_layer)

        i_net_layer = i_net_layer + 1
        net:add(mrf_module)
        next_mrf_idx = next_mrf_idx + 1 

        return true
      end

      --------------------------------------------------------------------------------------------------------
      -- local function for adding a content layer
      --------------------------------------------------------------------------------------------------------
      local function add_layer_content()
        i_content_layer = i_content_layer + 1
        table.insert(content_layers, i_content_layer, i_net_layer)   
        local target = nil
        if params.content_block_flag == 1 then
          target = net:forward(ini_block_caffe):clone() -- generate a fake block target 
        else
          target = net:forward(content_image_caffe):clone() -- generate the content target using content image
        end
        local norm = params.normalize_gradients
        local loss_module = nn.ContentLoss(params.content_weight, target, norm):float()
        if params.gpu >= 0 then
          loss_module:cuda()
        end
        i_net_layer = i_net_layer + 1
        net:add(loss_module)
        table.insert(content_losses, loss_module)
        next_content_idx = next_content_idx + 1      
      end

      --------------------------------------------------------------------------------------------------------
      -- local function for adding a style layer
      --------------------------------------------------------------------------------------------------------
      local function add_layer_style()
        i_style_layer = i_style_layer + 1
        table.insert(style_layers, i_style_layer, i_net_layer)
        local gram = GramMatrix():float()
        if params.gpu >= 0 then
          gram = gram:cuda()
        end
        local target_features = net:forward(style_image_caffe):clone()
        local target = gram:forward(target_features)
        target:div(target_features:nElement())

        local weight = params.style_weight * style_layer_weights[next_style_idx]
        local norm = params.normalize_gradients
        local loss_module = nn.StyleLoss(weight, target, norm):float()
        if params.gpu >= 0 then
          loss_module:cuda()
        end
        i_net_layer = i_net_layer + 1
        net:add(loss_module)
        table.insert(style_losses, loss_module)
        next_style_idx = next_style_idx + 1     
      end

      --------------------------------------------------------------------------------------------------------
      -- local function for printing inter-mediate result
      --------------------------------------------------------------------------------------------------------
      local function maybe_print(t, loss)
         local verbose = (params.print_iter > 0 and t % params.print_iter == 0)
         if verbose then
            print(string.format('Iteration %d / %d', t, params.num_iterations))
            for i, loss_module in ipairs(style_losses) do
               print(string.format('  Style %d loss: %f', i, loss_module.loss))
            end
         end
      end

      --------------------------------------------------------------------------------------------------------
      -- local function for saving inter-mediate result
      --------------------------------------------------------------------------------------------------------
      local function maybe_save(t)
         local should_save = params.save_iter > 0 and t % params.save_iter == 0
         should_save = should_save or t == params.num_iterations
         if should_save then
            local disp = deprocess(input_caffe:float())
            disp = image.minmax{tensor=disp, min=0, max=1}
            disp = image.scale(disp, params.render_size, 'bilinear')
            local filename = build_filename(params.output_image, t)
            filename = params.output_folder .. '/' .. 'res_' .. string.format('%d', params.res) .. '_' .. filename
            if t == params.num_iterations then
               filename = params.output_image
            end
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
         -- grad:cmul(mask)
         -- print(grad)
         local loss = 0
         for _, mod in ipairs(content_losses) do
            loss = loss + mod.loss
         end
         for _, mod in ipairs(style_losses) do
            loss = loss + mod.loss
         end
         maybe_print(num_calls, loss)
         maybe_save(num_calls)

         collectgarbage()
         -- optim.lbfgs expects a vector for gradients
         return loss, grad:view(grad:nElement())
      end

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

      ---------------------------------------------------
      -- add a pixel based mrf layer if necessary
      --------------------------------------------------- 
      if #mrf_layers_pretrained > 0 then
        if mrf_layers_pretrained[1] == 0 then
          add_layer_mrf()
        end
      end

      ---------------------------------------------------
      -- add a pixel based contentloss layer if necessary
      --------------------------------------------------- 
      if #content_layers_pretrained > 0 then
        if content_layers_pretrained[1] == 0 then
          add_layer_content()
        end
      end

      for i = 1, #cnn do
        if next_content_idx <= #content_layers_pretrained or next_style_idx <= #style_layers_pretrained or next_mrf_idx <= #mrf_layers_pretrained then

          local layer = cnn:get(i)

          i_net_layer = i_net_layer + 1
          net:add(layer)

          -- -- -- add mrf_losses layer
          if i == mrf_layers_pretrained[next_mrf_idx] then

           if add_layer_mrf() then
           else
              mrf_layer_patch_size[next_mrf_idx] = mrf_layer_patch_size[next_mrf_idx] - 1
                if add_layer_mrf() then
                else
                  mrf_layer_patch_size[next_mrf_idx] = mrf_layer_patch_size[next_mrf_idx] - 1
                  if add_layer_mrf() then
                  else
                    mrf_layer_patch_size[next_mrf_idx] = mrf_layer_patch_size[next_mrf_idx] - 1
                    if add_layer_mrf() then
                    else
                      print('error in add_layer')
                      do return end
                    end
                  end
                end
            end
          end

          -- add a content_losses layer
          if i == content_layers_pretrained[next_content_idx] then
            add_layer_content()
          end

          -- add style_losses layer
          if i == style_layers_pretrained[next_style_idx] then
            add_layer_style()
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

      print('style_layers: ')
        for i = 1, #style_layers do
        print(style_layers[i])
      end

      print('mrf_layers: ')
        for i = 1, #mrf_layers do
        print(mrf_layers[i])
      end

      -----------------------------------------------------
      print('Synthesis')
      -----------------------------------------------------
      local optim_state = {
        maxIter = params.num_iterations,
        nCorrection = params.nCorrection,
        verbose=true,
        tolX = 0,
        tolFun = 0,
      }        

      if params.init == 'random' then
         output_caffe = torch.randn(content_image_caffe:size()):float():mul(0.001)
         output = deprocess(output_caffe:float())
         print('random initialization ...')
      elseif params.init == 'image' then
         output_caffe = ini_image_caffe:clone():float()
         output = deprocess(output_caffe:float())
      else
         error('Invalid init type')
      end
      if params.gpu >= 0 then
         output_caffe = output_caffe:cuda()
      end
      input_caffe = output_caffe  
      mask = torch.Tensor(input_caffe:size()):fill(1)
      if params.gpu >= 0 then
         mask = mask:cuda()
      end   

      -- Run optimization.
      y = net:forward(input_caffe)
      dy = input_caffe.new(#y):zero()

      local x, losses = mylbfgs(feval, input_caffe, optim_state, nil, mask) 

      -- -- save the final output of this scale
      output = deprocess(input_caffe:float())
      output = image.minmax{tensor=output, min=0, max=1}
      local filename_output = params.output_folder .. '/' .. params.content_name .. '_to_' .. params.style_name .. '_MRF_res_' .. string.format('%d', params.res) .. '.png'
      image.save(filename_output, output) 

      local filename_temp = params.output_folder .. '/' .. 'syn_res_' .. string.format('%d', params.res) .. '.png'
      image.save(filename_temp, output) 

      net = nil
      ini_block_caffe = nil
      coord_block_y = nil
      coord_block_x = nil
      content_layers = nil
      i_content_layer = nil
      next_content_idx = nil
      style_layers = nil
      i_style_layer = nil
      next_style_idx = nil
      mrf_layers = nil
      i_mrf_layer = nil
      next_mrf_idx = nil
      content_losses, style_losses = nil, nil
      input_caffe = nil
      output_caffe = nil
      output = nil 
      num_calls = nil
      y = nil
      dy = nil
      mask = nil
      mask_weight = nil
      collectgarbage()
      local t_main = timer_main:time().real
      print('t_main: ' .. t_main .. ' seconds')
  end -- end of main



  function build_filename(output_image, iteration)
     local idx = string.find(output_image, '%.')
     local basename = string.sub(output_image, 1, idx - 1)
     local ext = string.sub(output_image, idx)
     return string.format('%s_%d%s', basename, iteration, ext)
  end

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
    img = img + mean_pixel
    local perm = torch.LongTensor{3, 2, 1}
    img = img:index(1, perm):div(256.0)
    return img
  end

  function computegrid(width, height, block_size, block_stride, flag_all)
     coord_block_y = torch.range(1, height - block_size + 1, block_stride) 
     if flag_all == 1 then
       if coord_block_y[#coord_block_y] < height - block_size + 1 then
          local tail = torch.Tensor(1)
          tail[1] = height - block_size + 1
          coord_block_y = torch.cat(coord_block_y, tail)
       end
     end

     coord_block_x = torch.range(1, width - block_size + 1, block_stride) 
     if flag_all == 1 then
       if coord_block_x[#coord_block_x] < width - block_size + 1 then
          local tail = torch.Tensor(1)
          tail[1] = width - block_size + 1
          coord_block_x = torch.cat(coord_block_x, tail)
       end
    end

     return coord_block_x, coord_block_y
  end

  function computeMRF(input, size, stride, gpu)

    local coord_x, coord_y = computegrid(input:size()[3], input:size()[2], size, stride)
    local dim_1 = input:size()[1] * size * size
    local dim_2 = coord_y:nElement()
    local dim_3 = coord_x:nElement()
    local t_feature_mrf = torch.Tensor(dim_2 * dim_3, input:size()[1], size, size)

    if gpu >= 0 then
      t_feature_mrf = t_feature_mrf:cuda()
    end

    local count = 1
    for i_row = 1, dim_2 do
      for i_col = 1, dim_3 do
        t_feature_mrf[count] = input[{{1, input:size()[1]}, {coord_y[i_row], coord_y[i_row] + size - 1}, {coord_x[i_col], coord_x[i_col] + size - 1}}]
        count = count + 1
      end
    end
    local feature_mrf = t_feature_mrf:reshape(dim_2 * dim_3, dim_1)

    return t_feature_mrf, feature_mrf, coord_x, coord_y
  end


  function computeBB(width, height, alpha)
    local min_x, min_y, max_x, max_y
    local x1 = 1
    local y1 = 1
    local x2 = width
    local y2 = 1
    local x3 = width
    local y3 = height
    local x4 = 1
    local y4 = height
    local x0 = width / 2
    local y0 = height / 2

    x1r = x0+(x1-x0)*math.cos(alpha)+(y1-y0)*math.sin(alpha)
    y1r = y0-(x1-x0)*math.sin(alpha)+(y1-y0)*math.cos(alpha)

    x2r = x0+(x2-x0)*math.cos(alpha)+(y2-y0)*math.sin(alpha)
    y2r = y0-(x2-x0)*math.sin(alpha)+(y2-y0)*math.cos(alpha)

    x3r = x0+(x3-x0)*math.cos(alpha)+(y3-y0)*math.sin(alpha)
    y3r = y0-(x3-x0)*math.sin(alpha)+(y3-y0)*math.cos(alpha)

    x4r = x0+(x4-x0)*math.cos(alpha)+(y4-y0)*math.sin(alpha)
    y4r = y0-(x4-x0)*math.sin(alpha)+(y4-y0)*math.cos(alpha)

    print(x1r .. ' ' .. y1r .. ' ' .. x2r .. ' ' .. y2r .. ' ' .. x3r .. ' ' .. y3r .. ' ' .. x4r .. ' ' .. y4r)
    if alpha > 0 then
      -- find intersection P of line [x1, y1]-[x4, y4] and [x1r, y1r]-[x2r, y2r]
      local px1 = ((x1 * y4 - y1 * x4) * (x1r - x2r) - (x1 - x4) * (x1r * y2r - y1r * x2r)) / ((x1 - x4) * (y1r - y2r) - (y1 - y4) * (x1r - x2r))
      local py1 = ((x1 * y4 - y1 * x4) * (y1r - y2r) - (y1 - y4) * (x1r * y2r - y1r * x2r)) / ((x1 - x4) * (y1r - y2r) - (y1 - y4) * (x1r - x2r))
      local px2 = px1 + 1
      local py2 = py1
      print(px1 .. ' ' .. py1)
      -- find the intersection Q of line [px1, py1]-[px2, py2] and [x2r, y2r]-[x3r][y3r]

      local qx = ((px1 * py2 - py1 * px2) * (x2r - x3r) - (px1 - px2) * (x2r * y3r - y2r * x3r)) / ((px1 - px2) * (y2r - y3r) - (py1 - py2) * (x2r - x3r))
      local qy = ((px1 * py2 - py1 * px2) * (y2r - y3r) - (py1 - py2) * (x2r * y3r - y2r * x3r)) / ((px1 - px2) * (y2r - y3r) - (py1 - py2) * (x2r - x3r))  
      print(qx .. ' ' .. qy)

      min_x = width - qx
      min_y = qy
      max_x = qx
      max_y = height - qy
    else if alpha < 0 then
      -- find intersection P of line [x2, y2]-[x3, y3] and [x1r, y1r]-[x2r, y2r]
      local px1 = ((x2 * y3 - y2 * x3) * (x1r - x2r) - (x2 - x3) * (x1r * y2r - y1r * x2r)) / ((x2 - x3) * (y1r - y2r) - (y2 - y3) * (x1r - x2r))
      local py1 = ((x2 * y3 - y1 * x3) * (y1r - y2r) - (y2 - y3) * (x1r * y2r - y1r * x2r)) / ((x2 - x3) * (y1r - y2r) - (y2 - y3) * (x1r - x2r))
      local px2 = px1 - 1
      local py2 = py1
      -- find the intersection Q of line [px1, py1]-[px2, py2] and [x1r, y1r]-[x4r][y4r]
      local qx = ((px1 * py2 - py1 * px2) * (x1r - x4r) - (px1 - px2) * (x1r * y4r - y1r * x4r)) / ((px1 - px2) * (y1r - y4r) - (py1 - py2) * (x1r - x4r))
      local qy = ((px1 * py2 - py1 * px2) * (y1r - y4r) - (py1 - py2) * (x1r * y4r - y1r * x4r)) / ((px1 - px2) * (y1r - y4r) - (py1 - py2) * (x1r - x4r))  
      min_x = qx
      min_y = qy
      max_x = width - min_x
      max_y = height - min_y
      else
        min_x = x1
        min_y = y1
        max_x = x2
        max_y = y3
      end
    end

    return math.floor(min_x), math.floor(min_y), math.floor(max_x), math.floor(max_y)
  end

  ------------------------------------------------------------------------
  -- excute main
  ------------------------------------------------------------------------
  

  -- input options
  cmd:option('-proto_file', 'data/models/VGG_ILSVRC_19_layers_deploy.prototxt')
  cmd:option('-model_file', 'data/models/VGG_ILSVRC_19_layers.caffemodel')
  cmd:option('-normalize_gradients', false)

  local ini_name = content_name

  local result_name = content_name .. '_to_' .. style_name .. '_MRF'
  cmd:option('-style_name', style_name, 'style name')
  cmd:option('-content_name', content_name, 'content name')
  cmd:option('-style_image', './data/style/' .. style_name .. '.jpg',
            'Style target image')
  cmd:option('-mrf_image', './data/style/' .. style_name .. '.jpg',
            'MRF target image')
  cmd:option('-content_image', './data/content/' .. content_name .. '.jpg',
            'Content target image')
  cmd:option('-ini_image', './data/content/' .. ini_name .. '.jpg',
            'initial target image')

  cmd:option('-init', ini_method, 'random|image')

  -- gpu options
  cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')
  cmd:option('-backend', 'cudnn', 'nn|cudnn')

  cmd:option('-content_layers_pretrained', {23}, '')
  cmd:option('-content_weight', content_weight)

  

  -- resolution options
  cmd:option('-render_size', 256, '')

  -- l-bfgs options
  cmd:option('-nCorrection', 100)

  -- Output options
  cmd:option('-print_iter', 50)
  cmd:option('-save_iter', 50)
  cmd:option('-output_image', 'out.png')

  cmd:option('-tv_weight', 1e-3)

  cmd:option('-style_layers_pretrained', {}, '')
  cmd:option('-style_layer_weights', {}, '')
  cmd:option('-style_weight', 1e2)

  cmd:option('-mrf_layers_pretrained', mrf_layers, '')
  cmd:option('-mrf_layer_weights', {3e-5, 3e-5, 3e-5, 3e-5, 3e-5, 3e-5, 3e-5}, '') 
  cmd:option('-mrf_layer_patch_size', mrf_patch_size, 'patch size')
  cmd:option('-mrf_layer_sample_stride', {1, 1, 1, 1, 1, 1, 1, 1}, 'stride for sampling mrf from style images, this could be make very sparse to save memoery & time')
  cmd:option('-mrf_layer_synthesis_stride', {1, 1, 1, 1, 1, 1, 1}, 'stride for synthesis mrf on the output image. In general this should be kept small so patches overlap')
  cmd:option('-mrf_layer_confidence_threshold', {0, 0, 0, 0, 0, 0}, 'threshold for adding mrf into the target. MRF with confidence smaller than this value will not be used')

  cmd:option('-mrf_num_rotation', mrf_num_rotation, '')
  cmd:option('-mrf_num_scale', mrf_num_scale, '')
  cmd:option('-mrf_step_rotation', math.pi/24, '')
  cmd:option('-mrf_step_scale', 1.05, '')

  cmd:option('-output_folder', 'data/result/trans/MRF/' .. result_name, 'output folder')
  os.execute("mkdir " .. 'data/result/trans/MRF/')

  cmd:option('-image_size', 64, 'Maximum height / width of generated image')
  cmd:option('-style_size', 64, 'Maximum height / width of style image')
  -- cmd:option('-style_size', 41, 'Maximum height / width of style image') -- 144
  -- cmd:option('-style_size', 41, 'Maximum height / width of style image')

  cmd:option('-render_size', 256, '')
  cmd:option('-res', 1, 'resolution of synthesis')
  cmd:option('-num_iterations', num_iter[1])

  local params = cmd:parse(arg)
  os.execute("mkdir " .. params.output_folder)

  ---------------------------------------------------------------
  -- Resolution 1
  ---------------------------------------------------------------
  main(params)

  ---------------------------------------------------------------
  -- Resolution 2
  ---------------------------------------------------------------
  cmd:option('-ini_image', params.output_folder .. '/' .. 'syn_res_1.png',
             'initial target image')
  cmd:option('-image_size', 128, 'Maximum height / width of generated image')
  cmd:option('-style_size', 128, 'Maximum height / width of style image')
  -- cmd:option('-style_size', 83, 'Maximum height / width of style image') -- 144
  -- cmd:option('-style_size', 82, 'Maximum height / width of style image')

  cmd:option('-render_size', 256, '')
  cmd:option('-init', 'image', 'random|image')
  cmd:option('-res', 2, 'resolution of synthesis')
  cmd:option('-num_iterations', num_iter[2])
  
  -- cmd:option('-mrf_layer_patch_size', {3, 3, 3, 3, 3, 3, 3}, 'patch size')
  local params = cmd:parse(arg)
  main(params)

  ---------------------------------------------------------------
  -- Resolution 3
  ---------------------------------------------------------------
  cmd:option('-ini_image', params.output_folder .. '/' .. 'syn_res_2.png',
             'initial target image')
  cmd:option('-image_size', 256, 'Maximum height / width of generated image')
  cmd:option('-style_size', 256, 'Maximum height / width of style image')
  -- cmd:option('-style_size', 166, 'Maximum height / width of style image') -- 144
  -- cmd:option('-style_size', 165, 'Maximum height / width of style image')

  cmd:option('-render_size', 256, '')
  cmd:option('-init', 'image', 'random|image')
  cmd:option('-res', 3, 'resolution of synthesis')
  cmd:option('-num_iterations', num_iter[3])

  -- use larger patch has the danger of crash gpu
  cmd:option('-mrf_layer_patch_size', {3, 3, 3, 3, 3, 3, 3}, 'patch size')

  local params = cmd:parse(arg)
  main(params)

  ---------------------------------------------------------------
  -- Resolution 4
  ---------------------------------------------------------------
  cmd:option('-ini_image', params.output_folder .. '/' .. 'syn_res_3.png',
             'initial target image')
  cmd:option('-image_size', 384, 'Maximum height / width of generated image')
  cmd:option('-style_size', 384, 'Maximum height / width of style image')
  -- cmd:option('-style_size', 249, 'Maximum height / width of style image') -- 144
  -- cmd:option('-style_size', 247, 'Maximum height / width of style image') 

  cmd:option('-render_size', 384, '')
  cmd:option('-init', 'image', 'random|image')
  cmd:option('-res', 4, 'resolution of synthesis')
  cmd:option('-num_iterations', num_iter[4])

  -- use larger patch has the danger of crash gpu
  -- cmd:option('-mrf_layer_patch_size', {2, 2, 2, 2, 2, 2, 2}, 'patch size')

  cmd:option('-mrf_layer_patch_size', {3, 3}, 'patch size')
  cmd:option('-mrf_layer_sample_stride', {2, 2}, 'stride for sampling mrf from style images, this could be make very sparse to save memoery & time')
  cmd:option('-mrf_layer_synthesis_stride', {2, 2}, 'stride for synthesis mrf on the output image. In general this should be kept small so patches overlap')


  local params = cmd:parse(arg)
  main(params)

  return flag_state
end

return {
  state = run_test
}
