local THNN = require 'nn.THNN'
local CrossbarLinearWvar, parent = torch.class('nn.CrossbarLinearWvar', 'nn.Module')


function CrossbarLinearWvar:__init(inputSize, outputSize, accumN)
   local delayedReset = self.reset
   self.reset = function() end
   parent.__init(self, inputSize, outputSize)
   self.reset = delayedReset
   self.weight = torch.Tensor(outputSize, inputSize)
   self.VarP = torch.Tensor(outputSize, inputSize)
   self.VarM = torch.Tensor(outputSize, inputSize)
   self.accumN = accumN or inputSize
   self:reset()
   -- should nil for serialization, the reset will still work
   self.reset = nil
end

function CrossbarLinearWvar:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-1, 1)
         end)
      end
   else
      self.weight:uniform(-1, 1)
   end

   return self
end

function CrossbarLinearWvar:updateOutput(input)
	-- update Output
	input.THNN.CrossbarLinearWvar_updateOutput(
		self.output:cdata(),
		input:cdata(),
		self.weight:cdata(),
		self.VarP:cdata(),
		self.VarM:cdata(),
		self.accumN)
	return self.output
end


function CrossbarLinearWvar:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end
