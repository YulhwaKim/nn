local THNN = require 'nn.THNN'
local CrossbarCompute, parent = torch.class('nn.CrossbarCompute', 'nn.Linear')


function CrossbarCompute:__init(inputSize, outputSize)
   local delayedReset = self.reset
   self.reset = function() end
   parent.__init(self, inputSize, outputSize)
   self.reset = delayedReset
   self.weight = torch.Tensor(outputSize, inputSize)
   self.weightB = torch.Tensor(outputSize, inputSize)
   self.weightOrg = torch.Tensor(outputSize, inputSize)
   self.bias = torch.Tensor(outputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self.gradBias = torch.Tensor(outputSize)
   self.accumN = inputSize
   self:reset()
   -- should nil for serialization, the reset will still work
   self.reset = nil
end

function CrossbarCompute:reset(stdv)
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
         self.bias[i] = torch.uniform(-stdv, stdv)
      end
   else
      self.weight:uniform(-1, 1)
      self.bias:uniform(-stdv, stdv)
   end

   return self
end


function CrossbarCompute:binarized()
	self.weightOrg:copy(self.weight)
	self.weightB:copy(self.weight):add(1):div(2):clamp(0,1)
	self.weightB:round():mul(2):add(-1)
	return  self.weightB
end

function CrossbarCompute:updateOutput(input)
	-- get binary weight
	self.weightB = self:binarized()
	self.weight:copy(self.weightB)
	-- update Output
	input.THNN.CrossbarCompute_updateOutput(
		self.output:cdata(),
		input:cdata(),
		self.weight:cdata(),
		self.accumN)
	-- restore original weight
	self.weight:copy(self.weightOrg);
	return self.output
end


function CrossbarCompute:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end
