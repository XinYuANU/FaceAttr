require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'

local URnet = {}

function rmsprop(opfunc, x, config, state)
-- opfunc: a closure function 
-- x: parameters of model
-- config/state: the configure/state of the model

	-- get/update state
	local config  = config or {}
	local state   = state or config
	local lr      = config.learningRate or 1e-2
	local alpha   = config.alpha or 0.9
	local epsilon = config.epsilon or 1e-8
	
	-- evaluate f(x) and df/dx
	local fx, dfdx = opfunc(x)
	if config.optimize == true then 
		if not state.m then
			state.m = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
			state.tmp = torch.Tensor():typeAs(x):resizeAs(dfdx)
		end
		
		-- calculate new mean squared values
		state.m:mul(alpha)
		state.m:addcmul(1.0-alpha, dfdx, dfdx)
		
		-- perform update
		state.tmp:sqrt(state.m):add(epsilon)
		
		-- only update when optimize is true
		
		if config.numUpdates < 10 then
			io.write('Learning rate is: ', lr/50.0, ' ')
			x:addcdiv(-lr/50.0, dfdx, state.tmp)
		elseif config.numUpdates < 30 then
			io.write('Learning rate is: ', lr/5.0, ' ')
			x:addcdiv(-lr/5.0, dfdx, state.tmp)
		else
			io.write('Learning rate is: ', lr, ' ')
			x:addcdiv(-lr, dfdx, state.tmp)
		end
	end
	
	config.numUpdates = config.numUpdates + 1
	return x, {fx} -- why fx need {}?
end


function URnet.train(dataset_LR, dataset_HR, N)
	
	model_G:training()
	epoch = epoch or 0
	local N = N or dataset_HR:size()[1]
	local time = sys.clock()
	local G_L16 = 0
	local G_L19 = 0

	-- do one epoch
	print('\n<trainer> on training set: ')
	print('<trainer> online epoch # ' .. epoch .. ' [Batchsize = ' .. opt.batchSize .. ', lr = ' .. sgdState_G.learningRate .. '\n')
	
	for t = 1,N,opt.batchSize do
		local HR_inputs = torch.Tensor(opt.batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
		local LR_inputs = torch.Tensor(opt.batchSize, opt.geometry[1], 16, 16)

		
		local fevalG = function(x)
			collectgarbage()
			if not torch.eq(x,parameters_G) then 
				parameters_G:copy(x)
			end
			
			gradParameters_G:zero()
			
			-- forward pass
			local samples = model_G:forward(LR_inputs)
			local err_g   = criterion_G:forward(samples, HR_inputs)
			G_L16 = G_L16 + err_g
			
			-- backward pass
			--local df_samples_L16 = criterion_G:backward(samples, HR_inputs)
			local df_samples_L16 = criterion_G:backward(samples, HR_inputs)
			model_G:backward(LR_inputs, df_samples_L16)
			
			print('gradParameters_G: ', gradParameters_G:norm())
			return err_g, gradParameters_G
		end	
		
		local k = 1
		for i = t, t+opt.batchSize-1 do 
			local sample_HR = dataset_HR[IDX[i]]
			local sample_LR = dataset_LR[IDX[i]]
			HR_inputs[k] = sample_HR:clone()
			LR_inputs[k] = sample_LR:clone()
			k = k+1
		end
		rmsprop(fevalG, parameters_G, sgdState_G)
		
		xlua.progress(t, dataset_HR:size()[1])
	end
	
	time = sys.clock() - time
	time = time / dataset_HR:size()[1]
	print('<trainer> time to learn 1 sample = ' .. (time*1000) .. 'ms' .. 'gpu: ' .. opt.gpu)
	-- trainLogger:add{['% MSE accuarcy1'] = G_L16,['% MSE accuarcy2'] = G_L19}
	trainLogger:add{['MSE accuarcy1'] = G_L16/opt.batchSize}
	
	if epoch % opt.saveFreq == 0 then 
		local filename = paths.concat(opt.save, 'UR.net')
		local model_all = paths.concat(opt.save, 'model.net')
		os.execute('mkdir -p ' .. sys.dirname(filename))
		if paths.filep(filename) then
			os.execute('mv ' .. model_all .. ' ' .. model_all .. '.old')
		end
		print('<trainer> saving network to ' ..filename)
		torch.save(filename, {parameters_G})
		torch.save(model_all, {model_G}) -- added on April 10
	end
	epoch = epoch + 1
end
return  URnet
