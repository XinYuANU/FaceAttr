require 'torch'
require 'nn'
require 'cunn'
require 'optim'
require 'pl'

local URnet = {}


function rmsprop(opfunc, x, config, state)
	
    -- (0) get/update state
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 1e-2
    local alpha = config.alpha or 0.9
    local epsilon = config.epsilon or 1e-8

    -- (1) evaluate f(x) and df/dx
    local fx, dfdx = opfunc(x)
    if config.optimize == true then
        -- (2) initialize mean square values and square gradient storage
        if not state.m then
          state.m = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
          state.tmp = torch.Tensor():typeAs(x):resizeAs(dfdx)
        end

        -- (3) calculate new (leaky) mean squared values
        state.m:mul(alpha)
        state.m:addcmul(1.0-alpha, dfdx, dfdx)

        -- (4) perform update
        state.tmp:sqrt(state.m):add(epsilon)
        -- only opdate when optimize is true
        
        
		if config.numUpdates < 50 then
			  --io.write(" ", lr/50.0, " ")
			  x:addcdiv(-lr/50.0, dfdx, state.tmp)
		elseif config.numUpdates < 100 then
			--io.write(" ", lr/5.0, " ")
			x:addcdiv(-lr /5.0, dfdx, state.tmp)
		else 
		  --io.write(" ", lr, " ")
		  x:addcdiv(-lr, dfdx, state.tmp)
		end
    end
    config.numUpdates = config.numUpdates +1
  

    -- return x*, f(x) before optimization
    return x, {fx}
end


function adam(opfunc, x, config, state)
    --print('ADAM')
    -- (0) get/update state
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 0.001

    local beta1 = config.beta1 or 0.9
    local beta2 = config.beta2 or 0.999
    local epsilon = config.epsilon or 1e-8

    -- (1) evaluate f(x) and df/dx
    local fx, dfdx = opfunc(x)
    if config.optimize == true then
	    -- Initialization
	    state.t = state.t or 0
	    -- Exponential moving average of gradient values
	    state.m = state.m or x.new(dfdx:size()):zero()
	    -- Exponential moving average of squared gradient values
	    state.v = state.v or x.new(dfdx:size()):zero()
	    -- A tmp tensor to hold the sqrt(v) + epsilon
	    state.denom = state.denom or x.new(dfdx:size()):zero()

	    state.t = state.t + 1
	    
	    -- Decay the first and second moment running average coefficient
	    state.m:mul(beta1):add(1-beta1, dfdx)
	    state.v:mul(beta2):addcmul(1-beta2, dfdx, dfdx)

	    state.denom:copy(state.v):sqrt():add(epsilon)

	    local biasCorrection1 = 1 - beta1^state.t
	    local biasCorrection2 = 1 - beta2^state.t
	    
		local fac = 1
		if config.numUpdates < 10 then
		    fac = 50.0
		elseif config.numUpdates < 30 then
		    fac = 5.0
		else 
		    fac = 1.0
		end
		io.write(" ", lr/fac, " ")
        local stepSize = (lr/fac) * math.sqrt(biasCorrection2)/biasCorrection1
	    -- (2) update x
	    x:addcdiv(-stepSize, state.m, state.denom)
    end
    config.numUpdates = config.numUpdates +1
    -- return x*, f(x) before optimization
    return x, {fx}
end


-- training function

function URnet.train(dataset_HR, dataset_LR_HDF5, dataset_attr)

	model_G:training()
	epoch = epoch or 0
	local N = N or ntrain
	local dataBatchSize = opt.batchSize / 2
	local time = sys.clock()
	local err_gen = 0
  
    local inputs = torch.Tensor(opt.batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
    local targets = torch.Tensor(opt.batchSize)
	
	local inputs_HR = torch.Tensor(opt.batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
	local inputs_LR = torch.Tensor(opt.batchSize, opt.geometry[1], 16, 16)
	
	-- do one epoch
	print('\n<trainer> on training set:')
	print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ' lr = ' .. sgdState_G.learningRate .. ']')
	for t = 1,N,opt.batchSize do --dataBatchSize do

	----------------------------------------------------------------------
	-- create closure to evaluate f(X) and df/dX of discriminator

		----------------------------------------------------------------------
		-- create closure to evaluate f(X) and df/dX of generator 
		local fevalG = function(x)
		  collectgarbage()
		  if x ~= parameters_G then -- get new parameters
			parameters_G:copy(x)
		  end
		  
		  gradParameters_G:zero() -- reset gradients

		  -- forward pass
		  local samples = model_G:forward(inputs_LR)
		  local g       = criterion_G:forward(samples, inputs_HR) 
		  err_gen       = err_gen + g
--		  local samples_percep = preprocess_image(samples:clone())
--		  local inputs_HR_percep = preprocess_image(inputs_HR:clone())  
--		  local err_percep = PerceptionLoss:forward(samples_percep, inputs_HR_percep)

		  --io.write("G:",f+g, " G:", tostring(sgdState_G.optimize)," D:",tostring(sgdState_D.optimize)," ", sgdState_G.numUpdates, " ", sgdState_D.numUpdates , "\n")
		  --io.flush()

		  --  backward pass
		
		  local df_G_samples = criterion_G:backward(samples, inputs_HR)   ---added by xin
--		  local df_vgg = PerceptionLoss:backward(samples_percep, inputs_HR_percep)   
		  local df_do = df_G_samples --+ df_vgg * opt.eta
		  model_G:backward(inputs_LR, df_do)

		--      print('gradParameters_G', gradParameters_G:norm())
		  return f,gradParameters_G
		end

	----------------------------------------------------------------------
	----------------------------------------------------------------------
	-- (2) Update G network: maximize log(D(G(z)))
	-- noise_inputs:normal(0, 1)

		local sample_HR = dataset_HR:index(IDX[{{t, t+opt.batchSize-1}}])
		sample_HR:mul(2):add(-1)
		sample_HR = sample_HR:index(2, torch.LongTensor{3,2,1})

		local k = 1  
		for i = t, t+opt.batchSize-1 do
		  inputs_HR[k] = sample_HR[k]:clone()
		  inputs_LR[k] = dataset_LR_HDF5[IDX[i]]:clone()
		  k = k+1  
		end
		targets:fill(1)
		rmsprop(fevalG, parameters_G, sgdState_G)
		
		nIter = nIter + 1
		if (nIter-1) % opt.saveFreq == 0 then
			netname = string.format('UR_net_%s',nIter)
			local filename = paths.concat(opt.save, netname)    
			os.execute('mkdir -p ' .. sys.dirname(filename))
			print('<trainer> saving network to '..filename)
			model_G:clearState()  
			torch.save(filename, {G = model_G})
			
			local to_plot = getSamples_compare(valData_LR, dataset_HR, dataset_attr, 50)
			torch.setdefaulttensortype('torch.FloatTensor')

			local formatted = image.toDisplayTensor({input = to_plot, nrow = 10})
			formatted:float()
			formatted = formatted:index(1,torch.LongTensor{3,2,1})

			image.save(opt.save .. '/UR_example_' .. (nIter or 0) .. '.png', formatted)
			torch.setdefaulttensortype('torch.CudaTensor')
		end
		
		-- display progress
		xlua.progress(t, ntrain)
	end -- end for loop over dataset

    -- time taken
    time = sys.clock() - time
    time = time / ntrain
    print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

    -- print confusion matrix
    --print(confusion)
    trainLogger:add{['MSE accuarcy1'] = err_gen/(ntrain/opt.batchSize)}

    -- next epoch
    epoch = epoch + 1


end

return URnet