require 'cunn-rtc'

local mytester = torch.Tester()

local cunntest = torch.TestSuite()
local precision_forward = 1e-4
local precision_backward = 1e-2
local times = {}
local nloop = 100

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

function cunntest.BCECriterionRTC_forward()
  local size = math.random(1,100)
  size = 256 * 2 * 20 * 28
  -- local size = math.random(1,100)
  local input = torch.rand(size):sigmoid()
  local target = torch.randn(size):gt(0):double()

  local tm = {}
  local title = string.format('BCECriterionRTC.forward, Size: %d', size)
  times[title] = tm

  local crit = nn.BCECriterionRTC()
  local groundtruth= crit:forward(input, target)
  local a = torch.Timer()
  for i = 1,nloop do
     groundtruth = crit:forward(input, target)
  end
  tm.cpu = a:time().real

  input = input:cuda()
  target = target:cuda()
  local g_crit = nn.BCECriterionRTC():cuda()
  local rescuda = g_crit:forward(input, target)
  a:reset()
  for i = 1,nloop do
     rescuda = g_crit:forward(input, target)
  end
  cutorch.synchronize()
  tm.gpu = a:time().real
  local errorVal = rescuda - groundtruth
  mytester:assertlt(errorVal, precision_forward, 'error on state (forward) ')
end


function cunntest.BCECriterionRTC_backward()
   local size = math.random(1,100)
   local size = math.random(1,100)
   size = 256 * 2 * 20 * 28
   -- local size = math.random(1,100)
   local input = torch.rand(size):sigmoid()
   local target = torch.randn(size):gt(0):double()

   local tm = {}
   local title = string.format('BCECriterionRTC.backward, Size %d', size)
   times[title] = tm

   local crit = nn.BCECriterionRTC()
   crit:forward(input, target)
   local groundgrad = crit:backward(input, target)
   local a = torch.Timer()
   for i = 1,nloop do
      groundgrad = crit:backward(input, target)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   target = target:cuda()
   local g_crit = nn.BCECriterionRTC():cuda()
   g_crit:forward(input, target)
   local rescuda = g_crit:backward(input, target)
   a:reset()
   for i = 1,nloop do
      rescuda = g_crit:backward(input, target)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:double() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

mytester:add(cunntest)
mytester:run()

print_timing = true
print(times)

   if print_timing then
       print ''
       print ' ------------------------------------------------------------------------------------------------'
       print '|  Module                                                                          |  Speedup    |'
       print ' ------------------------------------------------------------------------------------------------'
       for module,tm in pairs(times) do
           local str = string.format('| %-80s | %4.2f        |', module, (tm.cpu / (tm.gpu or 1e6)))
           print(str)
       end
       print ' ------------------------------------------------------------------------------------------------'
   end
