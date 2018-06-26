require 'stn'
require 'image'

spanet=nn.Sequential()

local concat=nn.ConcatTable()

-- first branch is there to transpose inputs to BHWD, for the bilinear sampler
tranet=nn.Sequential()
tranet:add(nn.Identity())
tranet:add(nn.Transpose({2,3},{3,4}))

-- second branch is the localization network
local locnet = nn.Sequential()
--locnet:add(nn.SpatialContrastiveNormalization(3,image.gaussian(5)))
locnet:add(cudnn.SpatialConvolution(32,128,3,3,1,1,1,1))
locnet:add(cudnn.ReLU(true))
locnet:add(cudnn.SpatialMaxPooling(2,2))
locnet:add(cudnn.SpatialConvolution(128,20,3,3, 1,1,1,1))
locnet:add(cudnn.ReLU(true))
locnet:add(cudnn.SpatialMaxPooling(2,2))
locnet:add(cudnn.SpatialConvolution(20,20,3,3))
locnet:add(cudnn.ReLU(true))
locnet:add(nn.View(20*2*2))
locnet:add(nn.Linear(20*2*2,20))
locnet:add(cudnn.ReLU(true))

-- we initialize the output layer so it gives the identity transform
--Affine
--local outLayer = nn.Linear(20,6)
--outLayer.weight:fill(0)
--local bias = torch.FloatTensor(6):fill(0)
--bias[1]=1
--bias[5]=1
--outLayer.bias:copy(bias)
--locnet:add(outLayer)

---- there we generate the grids
--locnet:add(nn.View(2,3))

--Similarity
local outLayer = nn.Linear(20,4)
outLayer.weight:fill(0)
local bias = torch.FloatTensor(4):fill(0)
bias[1]=1
bias[2]=1
outLayer.bias:copy(bias)
locnet:add(outLayer)

locnet:add(nn.AffineTransformMatrixGenerator(true, true, true))
	
locnet:add(nn.AffineGridGeneratorBHWD(16,16))

-- we need a table input for the bilinear sampler, so we use concattable
concat:add(tranet)
concat:add(locnet)

spanet:add(concat)
spanet:add(nn.BilinearSamplerBHWD())

-- and we transpose back to standard BDHW format for subsequent processing by nn modules
spanet:add(nn.Transpose({3,4},{2,3}))

return spanet