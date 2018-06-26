require 'torch'
require 'cunn'
require 'nn'
require 'cudnn'
require 'nngraph'

utils = {}
ngf = 32
function utils.loadModel(filename)
	model_G = nn.Sequential()
	model_G:add(cudnn.SpatialConvolution(3, ngf, 4, 4, 2, 2, 1, 1))
	model_G:add(cudnn.SpatialBatchNormalization(ngf*1))
	model_G:add(nn.LeakyReLU(0.2, true))
	model_G:add(cudnn.SpatialConvolution(ngf, ngf*4, 4, 4, 2, 2, 1, 1))
	model_G:add(cudnn.SpatialBatchNormalization(ngf*4))
	model_G:add(nn.LeakyReLU(0.2, true))
	model_G:add(cudnn.SpatialConvolution(ngf*4, ngf*16, 4, 4, 2, 2, 1, 1))
	model_G:add(cudnn.SpatialBatchNormalization(ngf*16))
	model_G:add(nn.LeakyReLU(0.2, true))
	model_G:add(cudnn.SpatialConvolution(ngf*16, ngf*64, 2, 2, 1, 1))
	model_G:add(cudnn.SpatialBatchNormalization(ngf*64))
	model_G:add(nn.LeakyReLU(0.2, true))
	
	model_G:cuda()
	
	local model_table = torch.load(filename)
	model_in = model_table.G:cuda()
	
	num_model_in = #model_in
	num_model_G  = #model_G
	
	model_G = model_in:clone()
--	print(#model_G, #model_in)
--	print(model_G:get(1).weight[{{1}}], model_in:get(1).weight[{{1}}])
	
	while #model_G > num_model_G do
		model_G:remove(#model_G)
	end
	
	return model_G
	
end
return utils

--m = utils.loadModel('/media/admin-xinyu/4TB/xin/UR_FaceAttr/logs128_UR_AE/UR_net_127501')