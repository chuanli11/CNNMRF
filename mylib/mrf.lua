local MRFMM, parent = torch.class('nn.MRFMM', 'nn.Module')

function MRFMM:__init(weight, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, strength, threshold_conf, gpu, mrf_num_x, mrf_num_y)
   parent.__init(self)
   
   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH

   self.dW = dW
   self.dH = dH
   self.padW = padW or 0
   self.padH = padH or self.padW
   self.gpu = gpu
   self.weight_all = weight:clone() -- insteresting, has to use clone otherwise the result is inconsistent with cpu
   self.weight_reshapemag = torch.Tensor(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
   
   if self.gpu >= 0 then
     self.weight_reshapemag = self.weight_reshapemag:cuda()
   end

   for i_w = 1, self.nOutputPlane do
     self.weight_reshapemag[i_w] = self.weight_all[{{i_w, i_w}, {1, self.weight_all:size()[2]}}]:reshape(self.nInputPlane, self.kH, self.kW)
   end
   self.weight_reshapemag = self.weight_reshapemag * (self.nInputPlane * self.kH * self.kW)

   self.bias = torch.Tensor(nOutputPlane):fill(0)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane*kH*kW)
   self.gradBias = torch.Tensor(nOutputPlane)

   self.finput = torch.Tensor()
   self.fgradInput = torch.Tensor()
   
   self.strength = strength
   self.threshold_conf = threshold_conf
   self.weight_norm = torch.sqrt(torch.sum(torch.cmul(self.weight_all, self.weight_all), 2)):reshape(self.weight_all:size()[1], 1, 1)

   self.gradMRF = nil

   self.max_response = nil
   self.max_id = nil

   self.mrf_num_x = mrf_num_x
   self.mrf_num_y = mrf_num_y
end

local function makeContiguous(self, input, gradOutput)
   if not input:isContiguous() then
    print('not contiguous, make it so')
      self._input = self._input or input.new()
      self._input:resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput then
      if not gradOutput:isContiguous() then
   self._gradOutput = self._gradOutput or gradOutput.new()
   self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
   gradOutput = self._gradOutput
      end
   end
   return input, gradOutput
end

function MRFMM:updateOutput(input)
  local timer_ALL = torch.Timer()

   -- backward compatibility
   if self.padding then
      self.padW = self.padding
      self.padH = self.padding
      self.padding = nil
   end  

  input = makeContiguous(self, input)

  -- -- compute mrf of input 
  local timer_computemrf = torch.Timer()
  local t_mrf_input, mrf_input, coord_syn_x, coord_syn_y = computeMRF(input, self.kW, self.dW, self.gpu) 
  local t_computemrf = timer_computemrf:time().real

  local mrf_norm = torch.sqrt(torch.sum(torch.cmul(mrf_input, mrf_input), 2)):reshape(1, coord_syn_y:nElement(), coord_syn_x:nElement())
  local tensor_mrf_norm = torch.repeatTensor(mrf_norm, self.nOutputPlane, 1, 1) 
  if self.gpu >= 0 then
    tensor_mrf_norm = tensor_mrf_norm:cuda()
  end
  local tensor_weight_norm = torch.repeatTensor(self.weight_norm, 1, coord_syn_y:nElement(), coord_syn_x:nElement()) 

  local response = torch.Tensor(self.nOutputPlane, coord_syn_y:nElement(), coord_syn_x:nElement()) 
  if self.gpu >= 0 then
    response = response:cuda()
  end

  -- hacked up for memory safety
  local nOutputPlane_all = self.nOutputPlane
  local nOutputPlane_chunk = 512

  -- split all filters into chuncks
  local num_chunk = math.ceil(nOutputPlane_all / nOutputPlane_chunk) 
  for i_chunk = 1, num_chunk do
    -- processing each chunck
    local i_start = (i_chunk - 1) * nOutputPlane_chunk + 1
    local i_end = math.min(i_start + nOutputPlane_chunk - 1, nOutputPlane_all)
    self.weight = self.weight_all[{{i_start, i_end}, {1, self.weight_all:size()[2]}}]:clone()
    self.nOutputPlane = i_end - i_start + 1
    response[{{i_start, i_end}, {1, response:size()[2]}, {1, response:size()[3]}}] = input.nn.SpatialConvolutionMM_updateOutput(self, input)
  end

  if self.gpu >= 0 then
  else
    response = response:float()
    tensor_weight_norm = tensor_weight_norm:float()
  end

  response = response:cdiv(tensor_mrf_norm)
  response = response:cdiv(tensor_weight_norm)

  self.max_response, self.max_id = torch.max(response, 1)

  local timer_reconstruct = torch.Timer()

  self.gradMRF = input:clone() * 0

  -- least square 
  self.gradMRF_confident = input[1]:clone():fill(0) + 1e-10
  local i_mrf = 0
  for i_row = 1, coord_syn_y:nElement() do
  local i_row_syn = coord_syn_y[i_row]
    for i_col = 1, coord_syn_x:nElement() do 
      local i_col_syn = coord_syn_x[i_col]
      i_mrf = i_mrf + 1
      if self.max_response[1][i_row][i_col] >= self.threshold_conf then
        self.gradMRF[{{1, self.nInputPlane}, {i_row_syn, i_row_syn + self.kH - 1}, {i_col_syn, i_col_syn + self.kW - 1}}]:add(self.weight_reshapemag[self.max_id[1][i_row][i_col]] - t_mrf_input[i_mrf])
        self.gradMRF_confident[{{i_row_syn, i_row_syn + self.kH - 1}, {i_col_syn, i_col_syn + self.kW - 1}}]:add(1)    
      end
    end
  end  
  self.gradMRF:cdiv(torch.repeatTensor(self.gradMRF_confident, self.nInputPlane, 1, 1))

  local t_reconstruct = timer_reconstruct:time().real

  self.nOutputPlane = nOutputPlane_all

  self.output = input:clone()
  local t_all = timer_ALL:time().real
  print('t_all: ' .. t_all .. ' mrf: ' .. t_computemrf/t_all .. ' rec: ' .. t_reconstruct/t_all)

  t_mrf_input = nil
  mrf_input = nil
  mrf_norm = nil
  tensor_mrf_norm = nil
  tensor_weight_norm = nil
  response = nil
  collectgarbage()

  return self.output
end

function MRFMM:updateGradInput(input, gradOutput)

   if self.gradInput then
      input, self.target = makeContiguous(self, input, self.target)      
      if gradOutput:size()[1] == input:size()[1] then
        self.gradInput = gradOutput:clone() + self.gradMRF * self.strength * (-1)
      else
        self.gradInput = self.gradMRF * self.strength * (-1)
      end
      return self.gradInput
   end
end

function MRFMM:type(type)
   self.finput = torch.Tensor()
   self.fgradInput = torch.Tensor()
   return parent.type(self,type)
end

function MRFMM:__tostring__()
   local s = string.format('%s(%d -> %d, %dx%d', torch.type(self),
         self.nInputPlane, self.nOutputPlane, self.kW, self.kH)
   if self.dW ~= 1 or self.dH ~= 1 or self.padW ~= 0 or self.padH ~= 0 then
     s = s .. string.format(', %d,%d', self.dW, self.dH)
   end
   if (self.padW or self.padH) and (self.padW ~= 0 or self.padH ~= 0) then
     s = s .. ', ' .. self.padW .. ',' .. self.padH
   end
   return s .. ')'
end