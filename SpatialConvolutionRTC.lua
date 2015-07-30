local SpatialConvolution, parent = torch.class('nn.SpatialConvolutionRTC','nn.SpatialConvolutionMM')

function SpatialConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
  parent.__init(self,nInputPlane,nOutputPlane,kW,kH,dW,dH,padW,padH)
end

function SpatialConvolution:updateOutput(input)
  assert(input:nDimension() == 4)
  
  local inputWidth   = input:size(4)
  local inputHeight  = input:size(3)
  local outputWidth  = (inputWidth + 2*self.padW - self.kW) / self.dW + 1
  local outputHeight = (inputHeight + 2*self.padH - self.kH) / self.dH + 1

  local batchSize = input:size(1)

  self.output:resize(batchSize, self.nOutputPlane, outputHeight * outputWidth)

  local ones = self.fgradInput
  local columns = self.finput
  if ones:nDimension() ~= 2 or ones:size(1)*ones:size(2) < outputHeight*outputWidth then
    ones:resize(outputHeight * outputWidth,1):fill(1)
  end

  for i=1,batchSize do
    self.output[i]:t():mm(ones,self.bias:view(1,self.nOutputPlane))
    columns.nn.im2col(columns, input[i], self.kW, self.kH, self.dW, self.dH, self.padW, self.padH)
    self.output[i]:addmm(self.weight, columns)
  end

  self.output = self.output:view(batchSize, self.nOutputPlane, outputHeight, outputWidth)
end
