local THNN = require 'nn.THNN'
local CrossbarSpatialConvolution, parent = torch.class('nn.CrossbarSpatialConvolution', 'nn.Module')

function CrossbarSpatialConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, accumN, binarize)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH

   self.dW = dW
   self.dH = dH
   self.padW = padW or 0
   self.padH = padH or self.padW

   self.weight = torch.Tensor(nOutputPlane, nInputPlane*kH*kW)
   self.weightB = torch.Tensor(nOutputPlane, nInputPlane*kH*kW)
   self.weightOrg = torch.Tensor(nOutputPlane, nInputPlane*kH*kW)
   self.accumN = accumN or inputSize
   
   self.binarize = binarize or false 
	if (binarize and type(self.binarize ~= 'boolean') then
		error('binarize flag must be boolean')
	end
   
   self:reset()
end


function CrossbarSpatialConvolution:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
   end
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
   else
      self.weight:uniform(-stdv, stdv)
   end
end


function CrossbarSpatialConvolution:binarized()
	self.weightOrg:copy(self.weight)
	self.weightB:copy(self.weight):add(1):div(2):clamp(0,1)
	self.weightB:round():mul(2):add(-1)
	return  self.weightB
end


function CrossbarSpatialConvolution:updateOutput(input)
   -- get binary weight
   if self.binarize == true then
	self.weightB = self:binarized()
	self.weight:copy(self.weightB)
   end
   
   assert(input.THNN, torch.type(input)..'.THNN backend not imported')
   self.finput = self.finput or input.new()
   self.fgradInput = self.fgradInput or input.new()
   -- backward compatibility
   if self.padding then
      self.padW = self.padding
      self.padH = self.padding
      self.padding = nil
   end
   -- update output
   input.THNN.CrossbarSpatialConvolution_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.weight:cdata(),
      self.finput:cdata(),
      self.accumN,
      self.kW, self.kH,
      self.dW, self.dH,
      self.padW, self.padH
   )
   -- restore original weight
   if self.binarize == true then
	self.weight:copy(self.weightOrg);
   end
   return self.output
end



function CrossbarSpatialConvolution:__tostring__()
   local s = string.format('%s(%d -> %d, %dx%d', torch.type(self),
         self.nInputPlane, self.nOutputPlane, self.kW, self.kH)
   if self.dW ~= 1 or self.dH ~= 1 or self.padW ~= 0 or self.padH ~= 0 then
     s = s .. string.format(', %d,%d', self.dW, self.dH)
   end
   if (self.padW or self.padH) and (self.padW ~= 0 or self.padH ~= 0) then
     s = s .. ', ' .. self.padW .. ',' .. self.padH
   end
   return s .. ')'
end
