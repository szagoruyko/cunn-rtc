local Pointwise, parent = torch.class('nn.Pointwise', 'nn.Module')

function Pointwise:__init(forward_op, backward_op)
  parent.__init(self)
  self.forward_op = forward_op:gsub('output', 'x'):gsub('input', 'y')
  self.backward_op = backward_op:gsub('gradInput', 'x'):gsub('gradOutput', 'z')
end

function Pointwise:updateOutput(input)
  self.output:resizeAs(input):apply2(input, self.forward_op)
  return self.output
end

function Pointwise:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input)
  if (self.backward_op:find'output') then
    self.gradInput:apply3(self.output, gradOutput, self.backward_op:gsub('output', 'y'))
  elseif (self.backward_op:find'input') then
    self.gradInput:apply3(input, gradOutput, self.backward_op:gsub('input', 'y'))
  end
  return self.gradInput
end 
