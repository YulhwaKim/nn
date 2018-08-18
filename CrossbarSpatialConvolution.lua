local THNN = require 'nn.THNN'
local CrossbarSpatialConvolution, parent = torch.class('nn.CrossbarSpatialConvolution', 'nn.Module')

function SpatialConvolutionMM:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
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
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane*kH*kW)
   self.gradBias = torch.Tensor(nOutputPlane)

   self:reset()
end
