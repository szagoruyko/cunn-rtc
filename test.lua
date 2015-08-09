require 'cunn-rtc'

local mytester = torch.Tester()

local cunntest = {}
local precision_forward = 1e-4
local precision_backward = 1e-2
local times = {}
local nloop = 1

function cunntest.LeakyReLU_forward()
   local size = math.random(1,100)

   local tm = {}
   local title = string.format('LeakyReLU forward %d -> %d', size, size)
   times[title] = tm

   local input = torch.randn(size):float()
   local sconv = nn.Threshold(0):float()
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gconv = nn.LeakyReLU(0):cuda()
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function cunntest.LeakyReLU_backward()
   local size = math.random(1,100)

   local tm = {}
   local title = string.format('LeakyReLU.backward %d -> %d', size, size)
   times[title] = tm

   local input = torch.randn(size):float()
   local gradOutput = torch.randn(size):float()
   local sconv = nn.Threshold(0):float()
   sconv:forward(input)
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      groundgrad = sconv:backward(input, gradOutput)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   local gconv = nn.LeakyReLU(0):cuda()
   gconv:forward(input)
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:backward(input, gradOutput)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function cunntest.SpatialConvolution_forward_batch()
   local bs = math.random(1,32)
   local from = math.random(1,32)
   local to = math.random(1,64)
   local ki = math.random(1,15)
   local kj = math.random(1,15)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local input = torch.randn(bs,from,inj,ini):cuda()
   local sconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj):cuda()
   local groundtruth = sconv:forward(input)
   cutorch.synchronize()

   local gconv = nn.SpatialConvolutionRTC(from,to,ki,kj,si,sj):cuda()
   gconv.weight:copy(sconv.weight)
   gconv.bias:copy(sconv.bias)
   local rescuda = gconv:forward(input)
   cutorch.synchronize()
   local error = rescuda:float() - groundtruth:float()
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end


function cunntest.SpatialConvolution_backward_batch()
   local bs = math.random(1,32)
   local from = math.random(1,32)
   local to = math.random(1,64)
   local ki = math.random(1,15)
   local kj = math.random(1,15)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local scale = math.random()

   local input = torch.randn(bs,from,inj,ini):cuda()
   local gradOutput = torch.randn(bs,to,outj,outi):cuda()
   local sconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj):cuda()
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput, scale)
   cutorch.synchronize()
   local groundweight = sconv.gradWeight
   local groundbias = sconv.gradBias

   local gconv = nn.SpatialConvolutionRTC(from,to,ki,kj,si,sj):cuda()
   gconv.weight:copy(sconv.weight)
   gconv.bias:copy(sconv.bias)
   gconv:forward(input)

   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')

   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(input, gradOutput, scale)
   cutorch.synchronize()
   local weightcuda = gconv.gradWeight
   local biascuda = gconv.gradBias

   local error = rescuda:float() - groundgrad:float()
   local werror = weightcuda:float() - groundweight:float()
   local berror = biascuda:float() - groundbias:float()

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
   mytester:assertlt(berror:abs():max(), precision_backward, 'error on bias (backward) ')
end

mytester:add(cunntest)
mytester:run()
