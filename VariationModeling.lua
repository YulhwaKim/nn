local THNN = require 'nn.THNN'
local VariationModeling, parent = torch.class('nn.VariationModeling', 'nn.Module')


function VariationModeling:__init(accumN, ptable)
	parent.__init(self)
	self.accumN = accumN or 1
	self.ptable = ptable or torch.ones(self.accumN+1, 1)
-- 	self.ref = torch.Tensor(1,1) -- for debugging
end

function VariationModeling:updateOutput(input)
	-- update Output
	input.THNN.VariationModeling_updateOutput(
		self.output:cdata(),
		input:cdata(),
		self.ptable:cdata(),
		self.accumN)
-- 		self.ref:cdata())
	return self.output
end
