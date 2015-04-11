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

mytester:add(cunntest)
mytester:run()
