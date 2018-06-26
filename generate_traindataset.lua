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
require 'stn'
require 'sys'
ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end

-- cannot get so large memory to save hdf5, so we only test every 2048 for 4 times
ntrain = 163904
num = 1024*8

GPU_ID = 1
cutorch.setDevice(GPU_ID) 

function saveImages(data, model, dataset_attr, foldername,start)
  local N = data:size(1)

  local inputs_lr = torch.Tensor(N,3,16,16)
  
  for i = 1,N do
	inputs_lr[i]  = data[i]
  end
  attr = dataset_attr[{{start+1,start+N}}]
  
  --sys.tic()
  local samples_UR = model:forward({inputs_lr:cuda(), attr:cuda()})
  --t = sys.toc()
  --print(t)  
  samples_UR = nn.HardTanh():forward(samples_UR)

  
  local to_plot = {}
  for i = 1,N do
    to_plot[i] = samples_UR[i]:float()
    torch.setdefaulttensortype('torch.FloatTensor')
    local GEN = image.toDisplayTensor({input=to_plot[i], nrow=1})
    --GEN:add(1):div(2):float()
    GEN = GEN:index(1,torch.LongTensor{3,2,1})
    
    filename = string.format("%06d.png",i+start)
    image.save(foldername .. filename, GEN)
  end  

  torch.setdefaulttensortype('torch.CudaTensor') 
  cutorch.setDevice(GPU_ID)  
   
end

torch.setdefaulttensortype('torch.CudaTensor')

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


local lowHd5 = hdf5.open('dataset/YTC_LR_unalign_30.hdf5', 'r')
local data_LR = lowHd5:read('YTC'):all()
data_LR:mul(2):add(-1)
lowHd5:close()
trainData_LR = data_LR[{{1, ntrain}}]
valData_LR = data_LR[{{ntrain+1, num+ntrain}}]

model = torch.load('logs128_UR_Attr_AE_Stack_Skip_perception_noTradeOff/adversarial_net_174001_old')
model_G = model.G
model_G:evaluate()

folder = 'dataset/HR_generated/'
if not paths.dirp(folder) then
	paths.mkdir(folder)
end

num_remainder = ntrain%100
num_loop      = (ntrain-num_remainder)/100
for i = 1,num_loop do
	saveImages(trainData_LR[{{(i-1)*100+1,i*100}}], model_G, dataset_Attr, folder, (i-1)*100)
end
if num_remainder ~= 0 then
	saveImages(trainData_LR[{{num_loop*100+1,ntrain}}], model_G, dataset_Attr, folder, num_loop*100)
end
