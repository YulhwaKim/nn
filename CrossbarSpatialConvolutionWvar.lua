local THNN = require 'nn.THNN'
local CrossbarSpatialConvolutionWvar, parent = torch.class('nn.CrossbarSpatialConvolutionWvar', 'nn.Module')

function CrossbarSpatialConvolutionWvar:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, accumN)
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
   self.VarP = torch.Tensor(nOutputPlane, nInputPlane*kH*kW)
   self.VarM = torch.Tensor(nOutputPlane, nInputPlane*kH*kW)
   self.accumN = accumN or inputSize
   
   self:reset()
end


function CrossbarSpatialConvolutionWvar:reset(stdv)
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


function CrossbarSpatialConvolutionWvar:updateOutput(input)
   assert(input.THNN, torch.type(input)..'.THNN backend not imported')
   self.finput = self.finput or input.new()
   -- backward compatibility
   if self.padding then
      self.padW = self.padding
      self.padH = self.padding
      self.padding = nil
   end
   -- update output
   input.THNN.CrossbarSpatialConvolutionWvar_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.weight:cdata(),
      self.finput:cdata(),
      self.VarP:cdata(),
      self.VarM:cdata(),
      self.accumN,
      self.kW, self.kH,
      self.dW, self.dH,
      self.padW, self.padH
   )
   return self.output
end



function CrossbarSpatialConvolutionWvar:__tostring__()
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
