local LeakyReLU, parent = torch.class('nn.LeakyReLU', 'nn.Pointwise')

function LeakyReLU:__init(p)
  assert(type(p) == 'number')
  local forward_op = 'output = input > 0 ? input : input *'..p
  local backward_op = 'gradInput = input > 0 ? gradOutput : gradOutput *'..p
  parent.__init(self, forward_op, backward_op)
end
