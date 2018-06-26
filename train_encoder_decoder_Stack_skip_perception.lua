------------------------------------------------------------
--- This code is based on the eyescream code released at
--- https://github.com/facebook/eyescream
--- If you find it usefull consider citing
--- http://arxiv.org/abs/1506.05751
------------------------------------------------------------
require 'hdf5'
require 'nngraph'
require 'torch'
require 'nn'
require 'cunn'
require 'optim'
require 'image'
require 'pl'
require 'paths'
require 'cudnn'
require 'threads'
require 'PerceptionLoss'
require 'preprocess'
require 'MultiBCECriterion_balance'
ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end
URnet = require 'adversarial_xin_Attr_AE_Stack_perception'
stn_L1 = require 'stn_L1_UpG'
stn_L2 = require 'stn_L2_UpG'

dl = require 'dataload'
----------------------------------------------------------------------
-- parse command-line options
opt = lapp[[
  -s,--save          (default "logs128_UR_Attr_AE_Stack_Skip_perception")      subdirectory to save logs
  --saveFreq         (default 5000)          save every saveFreq epochs
  -n,--network       (default "")          reload pretrained network
  -p,--plot                                plot while training
  -r,--learningRate  (default 0.001)        learning rate
  -b,--batchSize     (default 64)         batch size
  -m,--momentum      (default 0)           momentum, for SGD only
  --coefL1           (default 0)           L1 penalty on the weights
  --coefL2           (default 0)           L2 penalty on the weights
  -t,--threads       (default 4)           number of threads
  -g,--gpu           (default 0)           gpu to run on (default cpu)
  -d,--noiseDim      (default 512)         dimensionality of noise vector
  --K                (default 1)           number of iterations to optimize D for
  -w, --window       (default 3)           windsow id of sample image
  --scale            (default 128)          scale of images to train on
  --lambda           (default 0.01)       trade off D and Euclidean distance 
  --lambda_attr      (default 0.1)       trade off D_attr  
  --eta   	     (default 0.01)       trade off G and perception loss   
  --margin           (default 0.3)        trade off D and G
]]

if opt.margin <= 0 then 
	opt.save = opt.save .. '_noTradeOff'
end

if opt.gpu < 0 or opt.gpu > 3 then opt.gpu = false end

print(opt)
-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())
--[[
-- Dataset Loader 
--]]
-- load attributes
file = io.open('dataset/list_attr_celeba_selected.txt')
Attr_idx = {}
if file then
	for line in file:lines() do
		--print(line,type(line))		
		local tmp = string.gmatch(line, '%d')
		local attr = {}
		for c in tmp do
			table.insert(attr, c)
		end
		table.insert(Attr_idx, attr)
    end
end
file:close()
dataset_Attr = torch.squeeze(torch.FloatTensor{Attr_idx})
print("Dataset has " .. dataset_Attr:size(1) .. " Images for training")
--print(dataset_Attr:size())

ntrain = torch.floor(dataset_Attr:size(1)/opt.batchSize)*opt.batchSize
nval = 1024*8
ntrain = ntrain - nval
--ntrain = 29952
--nval = 1024

-- load image_dataset
datapath = 'dataset/'
loadsize = {3, opt.scale, opt.scale}
nthread  = opt.threads
-- train_dataset
train_imagefolder_HR = 'CelebA_Trim_HR'
train_filename_HR = 'filename_list.txt'
print('loading training_HR')
train_HR = dl.ImageClassPairs(datapath, train_filename_HR, train_imagefolder_HR, loadsize)

local lowHd5 = hdf5.open('dataset/YTC_LR_unalign_30.hdf5', 'r')
local data_LR = lowHd5:read('YTC'):all()
data_LR:mul(2):add(-1)
lowHd5:close()
trainData_LR = data_LR[{{1, ntrain}}]
valData_LR = data_LR[{{ntrain+1, nval+ntrain}}]

-- fix seed
torch.manualSeed(1)

if opt.gpu then
	cutorch.setDevice(opt.gpu + 1)
	print('<gpu> using device ' .. opt.gpu)
	torch.setdefaulttensortype('torch.CudaTensor')
else
	torch.setdefaulttensortype('torch.FloatTensor')
end

opt.geometry = {3, opt.scale, opt.scale}

ndf = 32
ngf = 32
nAttr = 18

model_D1 = nn.Sequential()
--model_D1:add(nn.Identity())
model_D1:add(cudnn.SpatialConvolution(3, ndf, 5, 5, 1, 1, 2, 2))
model_D1:add(cudnn.SpatialMaxPooling(2,2))  							-- 64*64
model_D1:add(cudnn.ReLU(true))
--model_D1:add(nn.SpatialDropout(0.2)) 
model_D1:add(cudnn.SpatialConvolution(ndf, 2*ndf, 5, 5, 1, 1, 2, 2))
model_D1:add(cudnn.SpatialMaxPooling(2,2))                       	-- 32*32 
model_D1:add(cudnn.ReLU(true))
--model_D1:add(nn.SpatialDropout(0.2)) 

model_D2 = nn.Sequential()
model_D2:add(nn.Replicate(32, 3, 4))
model_D2:add(nn.Replicate(32, 4, 4))

model_Parallel_D = nn.ParallelTable()
model_Parallel_D:add(model_D1)
model_Parallel_D:add(model_D2)

model_Join_D = nn.JoinTable(2, 4)

model_D = nn.Sequential()
model_D:add(model_Parallel_D)
model_D:add(model_Join_D)
  
--model_D:add(cudnn.SpatialConvolution(3+nAttr, ndf, 5, 5, 1, 1, 2, 2))
--model_D:add(cudnn.SpatialMaxPooling(2,2))  							-- 64*64
--model_D:add(cudnn.ReLU(true))
--model_D:add(nn.SpatialDropout(0.2)) 
--model_D:add(cudnn.SpatialConvolution(ndf, 2*ndf, 5, 5, 1, 1, 2, 2))
--model_D:add(cudnn.SpatialMaxPooling(2,2))                       	-- 32*32 
--model_D:add(cudnn.ReLU(true))
--model_D:add(nn.SpatialDropout(0.2)) 
model_D:add(cudnn.SpatialConvolution(2*ndf+nAttr, 4*ndf, 5, 5, 1, 1, 2, 2))
model_D:add(cudnn.SpatialMaxPooling(2,2))                            -- 16*16
model_D:add(cudnn.ReLU(true))
model_D:add(nn.SpatialDropout(0.2))  
model_D:add(cudnn.SpatialConvolution(4*ndf, 8*ndf, 3, 3, 1, 1, 1, 1))
model_D:add(cudnn.SpatialMaxPooling(2,2))                            -- 8*8
model_D:add(cudnn.ReLU(true))
model_D:add(nn.SpatialDropout(0.2))

model_D:add(nn.Reshape(8*8*8*ndf))
model_D:add(nn.Linear(8*8*8*ndf, 1024))
model_D:add(cudnn.ReLU(true))
model_D:add(nn.Dropout())
model_D:add(nn.Linear(1024, 1))
model_D:add(nn.Sigmoid())
	
----------------------------------------------------------------------	
local input1 = nn.Identity()()
local input2 = nn.Identity()()
local a1 = input2 - nn.View(-1, nAttr, 1, 1)

local e1 = input1 - cudnn.SpatialConvolution(3, ngf, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*1)  - nn.LeakyReLU(0.2, true)   -- 8
local e2 = e1  - cudnn.SpatialConvolution(ngf, ngf*4, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*4) - nn.LeakyReLU(0.2, true)  -- 4
local e3 = e2  - cudnn.SpatialConvolution(ngf*4, ngf*16, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*16)  - nn.LeakyReLU(0.2, true)-- 2
local e4 = e3  - cudnn.SpatialConvolution(ngf*16, ngf*64, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*64) - nn.LeakyReLU(0.2, true)  -- 1
local e5 = {e4, a1} - nn.JoinTable(2, 4)

local d2 = e5 - cudnn.SpatialFullConvolution(ngf*64+nAttr, ngf*32, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*32) - cudnn.ReLU(true)  -- 2
local d3 = {d2, e3} - nn.JoinTable(2, 4)
local d4 = d3 - cudnn.SpatialFullConvolution(ngf*48, ngf*24, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*24) - cudnn.ReLU(true)  -- 4
local d5 = {d4, e2} - nn.JoinTable(2, 4)
local d6 = d5 - cudnn.SpatialFullConvolution(ngf*28, ngf*16, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*16) - cudnn.ReLU(true)  -- 8
local d7 = {d6, e1} - nn.JoinTable(2, 4)
local d8 = d7 - cudnn.SpatialFullConvolution(ngf*17, ngf*8, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*8) - cudnn.ReLU(true)  -- 16

local d9 = d8 - nn.SpatialUpSamplingNearest(2) - stn_L1 - cudnn.SpatialConvolution(ndf*8, ndf* 4, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(ngf*4) - cudnn.ReLU(true) -- 32
local d10 = d9 - nn.SpatialUpSamplingNearest(2) - stn_L2 - cudnn.SpatialConvolution(ndf*4, ndf* 2, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(ngf*2) - cudnn.ReLU(true) -- 64
local d11 = d10 - nn.SpatialUpSamplingNearest(2) - cudnn.SpatialConvolution(ndf*2, ndf* 1, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(ngf*1) - cudnn.ReLU(true) -- 128

local d12 = d11 - cudnn.SpatialConvolution(ngf, 3, 5, 5, 1, 1, 2, 2)

model_G = nn.gModule({input1, input2}, {d12})

print('Copy model to gpu')
model_D:cuda()
model_G:cuda()  -- convert model to CUDA

-- loss function: negative log-likelihood
criterion_D = nn.BCECriterion():cuda()
criterion_G = nn.MSECriterion():cuda()

vgg_model = createVggmodel()
PerceptionLoss = nn.PerceptionLoss(vgg_model, 1):cuda()

-- retrieve parameters and gradients
parameters_D,gradParameters_D = model_D:getParameters()
parameters_G,gradParameters_G = model_G:getParameters()

-- print networks
print('Discriminator network:')
print(model_D)
print('Generator network:')
print(model_G)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
--testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- Training parameters
sgdState_D = {
	learningRate = opt.learningRate,
	momentum = opt.momentum,
	trade_off = false,
	optimize=true,
	numUpdates = 0,
	beta1 = 0.5
}
if opt.margin > 0 then
	sgdState_D.trade_off = true
end

sgdState_G = {
	learningRate = opt.learningRate,
	momentum = opt.momentum,
	optimize=true,
	numUpdates=0,
	beta1 = 0.5
}

function getSamples_compare_attrflip(dataset_LR, dataset_HR_fromFile, dataset_Attr, N)
	local N = N or 10
	
	dataset_HR = dataset_HR_fromFile:index(torch.range(1+ntrain, N+ntrain))
	dataset_HR:mul(2):add(-1)
	
	local inputs = torch.Tensor(N, 3, 16, 16)
	inputs:copy(dataset_LR[{{1, N}}])
	
	local inputs_attr = torch.Tensor(N, nAttr)
	inputs_attr:copy(dataset_Attr[{{ntrain+1, ntrain+N}}])
	--inputs_attr:mul(2):add(-1)
	
	local samples = model_G:forward({inputs:cuda(), inputs_attr:cuda()})
	samples = nn.HardTanh():forward(samples)
	
	local inputs_attr_flip = torch.FloatTensor(N, nAttr)
	inputs_attr_flip:fill(0)
	local samples_flip = model_G:forward({inputs:cuda(), inputs_attr_flip:cuda()})
	samples_flip = nn.HardTanh():forward(samples_flip)
	
	torch.setdefaulttensortype('torch.FloatTensor')
	dataset_HR = dataset_HR:index(2, torch.LongTensor{3,2,1})
	local to_plot = {}
	for i = 1,N do
		to_plot[#to_plot+1] = samples_flip[i]:float()
		to_plot[#to_plot+1] = samples[i]:float()
		to_plot[#to_plot+1] = dataset_HR[i]:float()
	end
	torch.setdefaulttensortype('torch.CudaTensor')
	return to_plot
end

nIter = 0

while true do 

	torch.setdefaulttensortype('torch.FloatTensor')
	
	trainLogger:style{['MSE accuarcy1'] = '-'}
	trainLogger:plot()
	
	IDX = torch.randperm(ntrain)
	IDX = IDX:long()
	
	if opt.gpu then 
		torch.setdefaulttensortype('torch.CudaTensor')
	else
		torch.setdefaulttensortype('torch.FloatTensor')
	end
	
	URnet.train(train_HR, trainData_LR, dataset_Attr)
		
	sgdState_D.momentum = math.min(sgdState_D.momentum + 0.0008, 0.7)
    sgdState_D.learningRate = math.max(opt.learningRate*0.95^epoch, 0.000001)
	
	sgdState_G.momentum = math.min(sgdState_G.momentum + 0.0008, 0.7)
	sgdState_G.learningRate = math.max(opt.learningRate*0.95^epoch, 0.000001)
	
	opt.lambda = math.max(opt.lambda*0.995, 0.005)   -- or 0.995
	--opt.lambda_attr = math.max(opt.lambda_attr*1.1, 0.5)   -- or 0.995
end
