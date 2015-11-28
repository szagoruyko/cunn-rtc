local ELU, parent = torch.class('nn.ELU', 'nn.Pointwise')

local fwd = 'output = input > 0 ? input : %f * (exp(input) - 1.f)'
local bwd = 'gradInput = output > 0 ? gradOutput : gradOutput * (output + %f)'

function ELU:__init(alpha)
  assert(type(alpha) == 'number')
  parent.__init(self, fwd:format(alpha), bwd:forward(alpha))
end
