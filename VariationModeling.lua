local THNN = require 'nn.THNN'
local VariationModeling, parent = torch.class('nn.VariationModeling', 'nn.Module')


function VariationModeling:__init(accumN, ptable, ip)
	parent.__init(self)
	self.accumN = accumN or 1
	self.ptable = ptable or torch.ones(self.accumN+1, 1)
-- 	self.ref = torch.Tensor(1,1) -- for debugging
	self. inplace = ip or false
		if (ip and type(ip) ~= 'boolean') then
			error('in-place flag must be boolean')
		end
end

function VariationModeling:updateOutput(input)
	-- check inplace
	if self.inplace then
		self.output:set(input)
	end
	-- update Output
	input.THNN.VariationModeling_updateOutput(
		self.output:cdata(),
		input:cdata(),
		self.ptable:cdata(),
		self.accumN)
-- 		self.ref:cdata())
	return self.output
end
