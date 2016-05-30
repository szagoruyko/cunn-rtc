local unfolds = require 'cunn-rtc.im2col'
local SpatialConvolution, parent = torch.class('nn.SpatialConvolutionRTC','nn.SpatialConvolutionMM')

function SpatialConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
  parent.__init(self,nInputPlane,nOutputPlane,kW,kH,dW,dH,padW,padH)
end

function SpatialConvolution:getIm2ColParams()
  return self.kW, self.kH, self.dW, self.dH, self.padW, self.padH
end

function SpatialConvolution:updateOutput(input)
  assert(input:nDimension() == 4)
  
  local outputWidth  = math.floor((input:size(4) + 2*self.padW - self.kW) / self.dW) + 1
  local outputHeight = math.floor((input:size(3) + 2*self.padH - self.kH) / self.dH) + 1

  self.output:resize(input:size(1), self.nOutputPlane, outputHeight, outputWidth)

  self.fgradInput = self.fgradInput or input.new()
  self.finput = self.finput or input.new()
  local ones = self.fgradInput
  local columns = self.finput
  if ones:nDimension() ~= 2 or ones:numel() ~= outputHeight*outputWidth then
    ones:resize(1,outputHeight * outputWidth):fill(1)
  end

  local o = self.output:view(input:size(1), self.nOutputPlane,-1)
  local bias = self.bias:view(self.nOutputPlane,1)

  for i=1,input:size(1) do
    unfolds.im2col(columns, input[i], self:getIm2ColParams())
    o[i]:mm(bias, ones):addmm(self.weight, columns)
  end

  return self.output
end

function SpatialConvolution:updateGradInput(input,gradOutput)
  assert(input:nDimension() == 4)
  assert(gradOutput:nDimension() == 4)

  self.gradInput:resizeAs(input)

  local columns = self.finput
  local go = gradOutput:view(input:size(1),self.nOutputPlane,-1)

  for i=1,input:size(1) do
    columns:mm(self.weight:t(), go[i])
    unfolds.col2im(self.gradInput[i], columns, self:getIm2ColParams())
  end
  return self.gradInput
end

function SpatialConvolution:accGradParameters(input, gradOutput, scale)
  assert(input:nDimension() == 4)
  assert(gradOutput:nDimension() == 4)

  local scale = scale or 1

  local outputWidth  = self.output:size(4)
  local outputHeight = self.output:size(3)

  local ones = self.fgradInput
  if ones:nDimension() ~= 2 or ones:numel() ~= outputHeight*outputWidth then
    ones:resize(1,outputHeight * outputWidth):fill(1)
  end

  local columns = self.finput
  local go = gradOutput:view(input:size(1), self.nOutputPlane,-1)

  for i=1,input:size(1) do
    unfolds.im2col(columns, input[i], self:getIm2ColParams())
    self.gradWeight:addmm(scale, go[i], columns:t())
    self.gradBias:addmv(scale, go[i], ones:view(-1)) 
  end
end

