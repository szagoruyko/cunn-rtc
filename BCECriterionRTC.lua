local BCECriterionRTC, parent = torch.class('nn.BCECriterionRTC', 'nn.BCECriterion')

local eps = '1e-12f'

local forward_op = ('x = - logf(x + %s) * y - logf(1.f - x + %s) * (1.f - y)'):format(eps, eps)
local backward_op = ('x = - (y - x) / ((1 - x + %s) * (x + %s))'):format(eps, eps)

function BCECriterionRTC:updateOutput(inputs, targets)
   if torch.type(inputs) == 'torch.CudaTensor' then
      self.buffer = self.buffer or inputs.new()
      self.buffer:resizeAs(inputs):copy(inputs)
      self.buffer:apply2(targets, forward_op)
      self.output = self.buffer:sum()
      if self.sizeAverage then
         self.output = self.output / inputs:numel()
      end
   else
      parent.updateOutput(self, inputs, targets)
   end
   return self.output
end

function BCECriterionRTC:updateGradInput(inputs, targets)
   if torch.type(inputs) == 'torch.CudaTensor' then
      self.gradInput:resizeAs(inputs):copy(inputs)
      self.gradInput:apply2(targets, backward_op)
      if self.sizeAverage then
         self.gradInput:div(inputs:numel())
      end
   else
      parent.updateGradInput(self, inputs, targets)
   end
   return self.gradInput
end
